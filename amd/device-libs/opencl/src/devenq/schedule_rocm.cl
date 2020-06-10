
#include "ockl_hsa.h"
#include "devenq.h"

typedef struct _SchedulerParam {
    ulong  kernarg_address;           //!< set to the VM address of SchedulerParam
    ulong  hidden_global_offset_x;    //!< set to 0 before queuing the scheduler
    ulong  hidden_global_offset_y;    //!< set to 0 before queuing the scheduler
    ulong  hidden_global_offset_z;    //!< set to 0 before queuing the scheduler
    ulong  thread_counter;            //!< set to 0 before queuing the scheduler
    __global hsa_queue_t* child_queue; //!< set to the device queue the child kernels will be queued to
    hsa_kernel_dispatch_packet_t scheduler_aql; //!< Dispatch packet used to relaunch the scheduler
    hsa_signal_t     complete_signal;  //!< Notify the host queue to continue processing
    __global AmdVQueueHeader* vqueue_header;  //!< The vqueue
    uint   signal;                   //!< Signal to stop the child queue
    uint   eng_clk;                  //!< Engine clock in Mhz
    __global AmdAqlWrap* parentAQL; //!< Host parent AmdAqlWrap packet
    ulong  write_index;              //!< Write Index to the child queue
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

static inline void
EnqueueDispatch(__global hsa_kernel_dispatch_packet_t* aqlPkt, __global SchedulerParam* param)
{
    __global hsa_queue_t* child_queue = param->child_queue;


    // ulong index = __ockl_hsa_queue_add_write_index(child_queue, 1, __ockl_memory_order_relaxed);
    // The original code seen above relies on PCIe 3 atomics, which might not be supported on some systems, so use a device side global
    // for workaround.
    ulong index = atomic_fetch_add_explicit((__global atomic_ulong*)&param->write_index, (ulong)1, memory_order_relaxed, memory_scope_device);

    const ulong queueMask = child_queue->size - 1;
    __global hsa_kernel_dispatch_packet_t* dispatch_packet = &(((__global hsa_kernel_dispatch_packet_t*)(child_queue->base_address))[index & queueMask]);
    *dispatch_packet = *aqlPkt;
}

static inline void
EnqueueScheduler(__global SchedulerParam* param)
{
    __global hsa_queue_t* child_queue = param->child_queue;

    // ulong index = __ockl_hsa_queue_add_write_index(child_queue, 1, __ockl_memory_order_relaxed);
    // The original code seen above relies on PCIe 3 atomics, which might not be supported on some systems, so use a device side global
    // for workaround.
    ulong index = atomic_fetch_add_explicit((__global atomic_ulong*)&param->write_index, (ulong)1, memory_order_relaxed, memory_scope_device);

    const ulong queueMask = child_queue->size - 1;
    __global hsa_kernel_dispatch_packet_t* dispatch_packet = &(((__global hsa_kernel_dispatch_packet_t*)(child_queue->base_address))[index & queueMask]);
    *dispatch_packet = param->scheduler_aql;

     // This is part of the PCIe 3 atomics workaround, to write the final write_index value back to the child_queue
    __ockl_hsa_queue_store_write_index(child_queue, index + 1, __ockl_memory_order_relaxed);

    __ockl_hsa_signal_store(child_queue->doorbell_signal, index, __ockl_memory_order_release);
}

void
__amd_scheduler_rocm(__global SchedulerParam* param)
{
    __global AmdVQueueHeader* queue = (__global AmdVQueueHeader*)(param->vqueue_header);
    __global AmdAqlWrap* wraps = (__global AmdAqlWrap*)&queue[1];
    __global uint* amask = (__global uint *)queue->aql_slot_mask;

    int launch = 0;
    int  grpId = get_group_id(0);
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
                    uint enqueueFlags = atomic_load_explicit( (__global atomic_uint*)(&disp->enqueue_flags), memory_order_relaxed, memory_scope_device);

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
                            EnqueueDispatch(&disp->aql, param);
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

    ulong threads_done = atomic_fetch_add_explicit((__global atomic_ulong*)&param->thread_counter, (ulong)1, memory_order_relaxed, memory_scope_device);
    if (threads_done >= (get_global_size(0) - 1)) {
        // The last thread finishes the processing
        __global AmdAqlWrap* hostParent = param->parentAQL;
        bool complete = atomic_load_explicit((__global atomic_uint*)&hostParent->child_counter, memory_order_relaxed, memory_scope_device) == 0;
        if (complete) {
            __ockl_hsa_signal_store(param->complete_signal, 0, __ockl_memory_order_relaxed);
        } else {
            param->thread_counter = 0;
            EnqueueScheduler(param);
        }
    }
}

