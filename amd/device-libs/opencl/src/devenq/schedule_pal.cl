
#include "devenq.h"

typedef struct _SchedulerParam {
    uint    signal;         //!< Signal to stop the child queue
    uint    eng_clk;        //!< Engine clock in Mhz
    ulong   hw_queue;       //!< Address to HW queue
    ulong   hsa_queue;      //!< Address to HSA dummy queue
    uint    useATC;         //!< GPU access to shader program by ATC.
    uint    scratchSize;    //!< Scratch buffer size
    ulong   scratch;        //!< GPU address to the scratch buffer
    uint    numMaxWaves;    //!< Num max waves on the asic
    uint    releaseHostCP;  //!< Releases CP on the host queue
    union {
        __global AmdAqlWrap* parentAQL;  //!< Host parent AmdAqlWrap packet
        ulong pad_parentAQL;
    };
    uint    dedicatedQueue; //!< Scheduler uses a dedicated queue
    uint    scratchOffset;  //!< Scratch buffer offset
    uint    ringGran64Dw ;  //!< WAVESIZE unit is 64 dwords instead of 256
    uint    reserved[1];    //!< Processed mask groups by one thread
} SchedulerParam;

static inline int
checkWaitEvents(__global AmdEvent** events, uint numEvents)
{
    for (uint i = 0; i < numEvents; ++i) {
        int status = atomic_load_explicit((__global atomic_uint*)(&events[i]->state), memory_order_relaxed, memory_scope_device);
        if (status != CL_COMPLETE)
            return status < 0 ? -1 : 0;
    }
    return 1;
}

static inline void
releaseEvent(__global AmdEvent* ev, __global uint* emask, __global AmdEvent* eb)
{
    uint c = atomic_fetch_sub_explicit((__global atomic_uint *)&ev->counter, 1U, memory_order_relaxed, memory_scope_device);
    if (c == 1U) {
        uint i = ev - eb;
        release_slot(emask, i);
    }
}

static inline void
releaseWaitEvents(__global AmdEvent** events, uint numEvents, __global uint* emask, __global AmdEvent* eb)
{
    for (uint i = 0; i < numEvents; ++i) {
        releaseEvent(events[i], emask, eb);
    }
}

static inline uint
min_command(uint slot_num, __global AmdAqlWrap* wraps)
{
    uint minCommand = 0xffffffff;
    for (uint idx = 0; idx < slot_num; ++idx) {
        __global AmdAqlWrap* disp = (__global AmdAqlWrap*)&wraps[idx];
        uint slotState = atomic_load_explicit((__global atomic_uint*)(&disp->state), memory_order_relaxed, memory_scope_device);
        if ((slotState != AQL_WRAP_FREE) && (slotState != AQL_WRAP_RESERVED)) {
            minCommand = min(disp->command_id, minCommand);
        }
    }
    return minCommand;
}

extern uint GetCmdTemplateHeaderSize(void);
extern uint GetCmdTemplateDispatchSize(void);
extern void EmptyCmdTemplateDispatch(ulong cmdBuf);
extern void RunCmdTemplateDispatch(
            ulong   cmdBuf,
            __global hsa_kernel_dispatch_packet_t* aqlPkt,
            ulong   scratch,
            ulong   hsaQueue,
            uint    scratchSize,
            uint    scratchOffset,
            uint    numMaxWaves,
            uint    useATC,
            uint    ringGran64Dw);

void
__amd_scheduler_pal(
    __global AmdVQueueHeader* queue,
    __global SchedulerParam* params,
    uint paramIdx)
{
    __global  SchedulerParam* param = &params[paramIdx];
    ulong hwDisp = param->hw_queue + GetCmdTemplateHeaderSize();
    __global AmdAqlWrap* hostParent = param->parentAQL;
    __global uint* counter = (__global uint*)(&hostParent->child_counter);
    __global uint* signal = (__global uint*)(&param->signal);
    __global AmdAqlWrap* wraps = (__global AmdAqlWrap*)&queue[1];
    __global uint* amask = (__global uint *)queue->aql_slot_mask;

    //! @todo This is an unexplained behavior.
    //! The scheduler can be launched one more time after termination.
    if (1 == atomic_load_explicit((__global atomic_uint*)&param->releaseHostCP,
        memory_order_acquire, memory_scope_device)) {
        return;
    }

    int launch = 0;
    int  grpId = get_group_id(0);
    hwDisp += GetCmdTemplateDispatchSize() * grpId;
    uint mskGrp = queue->mask_groups;

    for (uint m = 0; m < mskGrp && launch == 0; ++m) {
        uint maskId = grpId * mskGrp + m;
        uint mask = atomic_load_explicit((__global atomic_uint*)(&amask[maskId]), memory_order_relaxed, memory_scope_device);

        int baseIdx = maskId << 5;
        while (mask != 0) {
            uint sIdx = ctz(mask);
            uint idx = baseIdx + sIdx;
            mask &= ~(1 << sIdx);
            __global AmdAqlWrap* disp = (__global AmdAqlWrap*)&wraps[idx];
            uint slotState = atomic_load_explicit((__global atomic_uint*)(&disp->state), memory_order_acquire, memory_scope_device);
            __global AmdAqlWrap* parent = (__global AmdAqlWrap*)(disp->parent_wrap);
            __global AmdEvent* event = (__global AmdEvent*)(disp->completion);

            // Check if the current slot is ready for processing
            if (slotState == AQL_WRAP_READY) {
                if (launch == 0) {
                    // Attempt to find a new dispatch if nothing was launched yet
                    uint parentState = atomic_load_explicit((__global atomic_uint*)(&parent->state), memory_order_relaxed, memory_scope_device);
                    uint enqueueFlags = atomic_load_explicit((__global atomic_uint*)(&disp->enqueue_flags), memory_order_relaxed, memory_scope_device);

                    // Check the launch flags
                    if (((enqueueFlags == CLK_ENQUEUE_FLAGS_WAIT_KERNEL) ||
                        (enqueueFlags == CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP)) &&
                        (parentState != AQL_WRAP_DONE)) {
                        continue;
                    }

                    // Check if the wait list is COMPLETE
                    launch = checkWaitEvents((__global AmdEvent**)(disp->wait_list), disp->wait_num);

                    if (launch != 0) {
                        if (event != 0) {
                            event->timer[PROFILING_COMMAND_START] = ((ulong)__builtin_readcyclecounter() * (ulong)param->eng_clk) >> 10;
                        }
                        if (launch > 0) {
                            // Launch child kernel ....
                            RunCmdTemplateDispatch(hwDisp, &disp->aql, param->scratch, param->hsa_queue,
                                param->scratchSize, param->scratchOffset, param->numMaxWaves, param->useATC, param->ringGran64Dw);
                        } else if (event != 0) {
                            event->state = -1;
                        }
                        atomic_store_explicit((__global atomic_uint*)&disp->state, AQL_WRAP_BUSY, memory_order_relaxed, memory_scope_device);
                        releaseWaitEvents((__global AmdEvent**)(disp->wait_list), disp->wait_num, (__global uint*)queue->event_slot_mask,
                                          (__global AmdEvent*)queue->event_slots);
                        break;
                    }
                }
            } else if (slotState == AQL_WRAP_MARKER) {
                bool complete = false;
                if (disp->wait_num == 0) {
                    uint minCommand = min_command(queue->aql_slot_num, wraps);
                    complete = disp->command_id == minCommand;
                } else {
                    int status = checkWaitEvents((__global AmdEvent**)(disp->wait_list), disp->wait_num);
                    // Check if the wait list is COMPLETE
                    if (status != 0) {
                        complete = true;
                        releaseWaitEvents((__global AmdEvent**)(disp->wait_list), disp->wait_num, (__global uint*)queue->event_slot_mask,
                                          (__global AmdEvent*)queue->event_slots);
                        if (status < 0)
                            event->state = -1;
                    }
                }
                if (complete) {
                    // Decrement the child execution counter on the parent
                    atomic_fetch_sub_explicit((__global atomic_uint*)&parent->child_counter, 1, memory_order_relaxed, memory_scope_device);
                    if (event->state >= 0)
                        event->state = CL_COMPLETE;
                    atomic_store_explicit((__global atomic_uint*)&disp->state, AQL_WRAP_FREE, memory_order_relaxed, memory_scope_device);
                    release_slot(amask, idx);
                    releaseEvent(event, (__global uint*)queue->event_slot_mask, (__global AmdEvent*)queue->event_slots);
                }
            } else if ((slotState == AQL_WRAP_BUSY) || (slotState == AQL_WRAP_DONE)) {
                if (slotState == AQL_WRAP_BUSY) {
                    atomic_store_explicit((__global atomic_uint*)&disp->state, AQL_WRAP_DONE, memory_order_relaxed, memory_scope_device);
                    if (event != 0) {
                        event->timer[PROFILING_COMMAND_END] = ((ulong)__builtin_readcyclecounter() * (ulong)param->eng_clk) >> 10;
                    }
                }
                // Was CL_EVENT requested?
                if (event != 0) {
                    // The current dispatch doesn't have any outstanding children
                    if (disp->child_counter == 0) {
                        event->timer[PROFILING_COMMAND_COMPLETE] = ((ulong)__builtin_readcyclecounter() * (ulong)param->eng_clk) >> 10;
                        if (event->state >= 0) {
                            event->state = CL_COMPLETE;
                        }
                        if (event->capture_info != 0) {
                            __global ulong* values = (__global ulong*)event->capture_info;
                            values[0] = event->timer[PROFILING_COMMAND_END] - event->timer[PROFILING_COMMAND_START];
                            values[1] = event->timer[PROFILING_COMMAND_COMPLETE] - event->timer[PROFILING_COMMAND_START];
                        }
                        releaseEvent(event, (__global uint *)queue->event_slot_mask, (__global AmdEvent *)queue->event_slots);
                    }
                }
                // The current dispatch doesn't have any outstanding children
                if (disp->child_counter == 0) {
                    // Decrement the child execution counter on the parent
                    atomic_fetch_sub_explicit((__global atomic_uint*)&parent->child_counter, 1, memory_order_relaxed, memory_scope_device);
                    atomic_store_explicit((__global atomic_uint*)&disp->state, AQL_WRAP_FREE, memory_order_relaxed, memory_scope_device);
                    release_slot(amask, idx);
                }
            }
        }
    }

    if (launch <= 0) {
        EmptyCmdTemplateDispatch(hwDisp);
    }

    __global atomic_uint *againptr = param->dedicatedQueue ? (__global atomic_uint*)&param->signal : (__global atomic_uint*)&hostParent->child_counter;

    uint again = atomic_load_explicit(againptr, memory_order_relaxed, memory_scope_device);

    if (!again) {
        //! \todo Write deadcode to the template, but somehow
        //! the scheduler will be launched one more time.
        atomic_store_explicit((__global atomic_uint*)hwDisp, 0xdeadc0de, memory_order_relaxed, memory_scope_device);
        atomic_store_explicit((__global atomic_uint*)&param->signal, 0, memory_order_relaxed, memory_scope_device);
        atomic_store_explicit((__global atomic_uint*)&param->releaseHostCP, 1, memory_order_relaxed, memory_scope_device);
    }
}

