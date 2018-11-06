#include "as.h"
#include "ockl_hsa.h"

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define AC(P, E, V, O, R, S)                                                   \
    __opencl_atomic_compare_exchange_strong(P, E, V, O, R, S)
#define AL(P, O, S) __opencl_atomic_load(P, O, S)

typedef enum {
    __OCKL_AS_PACKET_EMPTY = 0,
    __OCKL_AS_PACKET_READY = 1
} __ockl_as_packet_type_t;

#define __OCKL_AS_PAYLOAD_ALIGNMENT 4
#define __OCKL_AS_PAYLOAD_BYTES 48

typedef enum {
    __OCKL_AS_PACKET_HEADER_TYPE = 0, // corresponds to HSA_PACKET_HEADER_TYPE
    __OCKL_AS_PACKET_HEADER_RESERVED0 = 8,
    __OCKL_AS_PACKET_HEADER_FLAGS = 13,
    __OCKL_AS_PACKET_HEADER_BYTES = 16,
    __OCKL_AS_PACKET_HEADER_SERVICE = 24,
} __ockl_as_packet_header_t;

typedef enum {
    __OCKL_AS_PACKET_HEADER_WIDTH_TYPE = 8,
    __OCKL_AS_PACKET_HEADER_WIDTH_RESERVED0 = 5,
    __OCKL_AS_PACKET_HEADER_WIDTH_FLAGS = 3,
    __OCKL_AS_PACKET_HEADER_WIDTH_BYTES = 8,
    __OCKL_AS_PACKET_HEADER_WIDTH_SERVICE = 8
} __ockl_as_packet_header_width_t;

// A packet is 64 bytes long, and the payload starts at index 16.
struct __ockl_as_packet_t {
    uint header;
    uint reserved1;
    ulong connection_id;

    uchar payload[__OCKL_AS_PAYLOAD_BYTES];
};

static uint
as_bitfield(uchar offset, uchar size, uint value)
{
    uint mask = (1 << size) - 1;
    value &= mask;
    return (uint)value << offset;
}

static uint
packet_type_as_bitfield(uchar value)
{
    return as_bitfield(__OCKL_AS_PACKET_HEADER_TYPE,
                       __OCKL_AS_PACKET_HEADER_WIDTH_TYPE, value);
}

static uint
packet_bytes_as_bitfield(uchar value)
{
    return as_bitfield(__OCKL_AS_PACKET_HEADER_BYTES,
                       __OCKL_AS_PACKET_HEADER_WIDTH_BYTES, value);
}

static uint
packet_service_as_bitfield(uchar value)
{
    return as_bitfield(__OCKL_AS_PACKET_HEADER_SERVICE,
                       __OCKL_AS_PACKET_HEADER_WIDTH_SERVICE, value);
}

static uint
packet_flags_as_bitfield(uchar value)
{
    return as_bitfield(__OCKL_AS_PACKET_HEADER_FLAGS,
                       __OCKL_AS_PACKET_HEADER_WIDTH_FLAGS, value);
}

static ulong
load_write_index(__global const __ockl_as_stream_t *stream,
                 __ockl_memory_order memory_order)
{
    return AL((__global const atomic_ulong *)&stream->write_index, memory_order,
              memory_scope_all_svm_devices);
}

static ulong
cas_write_index(__global __ockl_as_stream_t *stream, ulong expected,
                ulong value, __ockl_memory_order mem_order)
{
    ulong e = expected;
    AC((__global atomic_ulong *)&stream->write_index, &e, value, mem_order,
       memory_order_relaxed, memory_scope_all_svm_devices);
    return e;
}

static ulong
load_read_index(__global const __ockl_as_stream_t *stream,
                __ockl_memory_order memory_order)
{
    return __opencl_atomic_load((__global const atomic_ulong *)&stream
                                    ->read_index,
                                memory_order, memory_scope_all_svm_devices);
}

static void
signal_store_release(__ockl_as_signal_t signal, ulong packet_index)
{
    if (signal.handle == 0)
        return;
    hsa_signal_t hsa_sig = {signal.handle};
    __ockl_hsa_signal_store(hsa_sig, packet_index, __ockl_memory_order_release);
}

static __ockl_as_status_t
reserve_packet(__global __ockl_as_stream_t *stream,
               __private ulong *packet_index)
{
    ulong write_index = load_write_index(stream, __ockl_memory_order_acquire);
    ulong read_index = load_read_index(stream, __ockl_memory_order_acquire);

    if (stream->doorbell_signal.handle == 0 && write_index >= stream->size) {
        return __OCKL_AS_STATUS_OUT_OF_RESOURCES;
    }

    // Producers can only contend for packets within the window based
    // at read_index. This guarantees that a successful CAS returns a
    // packet that is immediately available for writing.
    //
    // The unsigned subtraction always works, even with
    // wraparound. But this should not matter in reality since 64-bit
    // indexes are not expected to ever reach wraparound.
    if (write_index - read_index >= stream->size) {
        return __OCKL_AS_STATUS_BUSY;
    }

    // Note that the CAS makes things painfully slow: even if multiple
    // packets are available, only one producer wins and then the rest
    // try all over again.
    ulong new_write_index = write_index + 1;
    if (cas_write_index(stream, write_index, new_write_index,
                        __ockl_memory_order_acq_rel) != write_index) {
        return __OCKL_AS_STATUS_BUSY;
    }

    *packet_index = write_index;
    return __OCKL_AS_STATUS_SUCCESS;
}

static __ockl_as_status_t
write_packet(__global __ockl_as_stream_t *stream, uchar service_id,
             ulong *connection_id, const uchar *str, uchar len, uchar flags)
{
    ulong packet_index;

    // Execute only one workitem at a time and return failure for the rest
    uint lane_id = __ockl_lane_u32();
    if (__builtin_amdgcn_readfirstlane(lane_id) != lane_id)
        return __OCKL_AS_STATUS_BUSY;

    __ockl_as_status_t status = reserve_packet(stream, &packet_index);

    if (status != __OCKL_AS_STATUS_SUCCESS)
        return status;

    __global __ockl_as_packet_t *packet =
        stream->base_address + packet_index % stream->size;

    if (flags & __OCKL_AS_CONNECTION_BEGIN) {
        *connection_id = packet_index;
    }
    packet->connection_id = *connection_id;

    // Bytes are written out only up to the capacity of the packet. The
    // rest are ignored.
    uchar m = __OCKL_AS_PAYLOAD_BYTES;
    len = len > m ? m : len;

    for (uchar ii = 0; ii != len; ++ii) {
        packet->payload[ii] = str[ii];
    }

    uint header = packet_flags_as_bitfield(flags);
    header |= packet_bytes_as_bitfield(len);
    header |= packet_service_as_bitfield(service_id);
    header |= packet_type_as_bitfield(__OCKL_AS_PACKET_READY);

    __opencl_atomic_store((__global atomic_uint *)&packet->header, header,
                          __ockl_memory_order_release,
                          memory_scope_all_svm_devices);

    signal_store_release(stream->doorbell_signal, packet_index);

    return __OCKL_AS_STATUS_SUCCESS;
}

__ockl_as_status_t
__ockl_as_write_block(__global __ockl_as_stream_t *stream, uchar service_id,
                      ulong *connection_id, const uchar *str, uint len,
                      uchar flags)
{
    // The block may be split into multiple packets. The BEGIN flag
    // should be only on the first packet, while the FLUSH/END flags
    // should be only on the last packet.
    //
    // Initialize flags for the first packet.
    uchar first = flags & __OCKL_AS_CONNECTION_BEGIN;
    uchar last = 0;

    int thread_done = 0;
    __ockl_as_status_t retval = __OCKL_AS_STATUS_UNKNOWN_ERROR;

    // FIXME: The funtion write_packet() uses a waterfall to ensure
    // that only one workitem succeeds at a time. But such an
    // arbitration over a shared resource assumes independent forward
    // progress, which is actually not guaranteed. The compiler moves
    // memory operations out of the waterfall; this is valid within
    // the memory consistency model, but it deadlocks the program. To
    // prevent this code motion, we use a wfall() loop wrapped around
    // the call to write_packet().
    //
    // The real issue here is the undefined behaviour in the absence
    // of independent forward progress, which will be fixed when we
    // eventually implement a trap handler. The protocol implemented
    // here will usually work, and is only meant as a prototype.
    do {
        if (thread_done == 0) {
            uchar plen = __OCKL_AS_PAYLOAD_BYTES;
            if (len <= plen) {
                // Determine flags for the last packet.
                last = flags &
                       (__OCKL_AS_CONNECTION_END | __OCKL_AS_CONNECTION_FLUSH);
                plen = len;
            }

            __ockl_as_status_t status =
                write_packet(stream, service_id, connection_id, str, plen,
                             first | last);

            switch (status) {
            case __OCKL_AS_STATUS_BUSY:
                break;
            case __OCKL_AS_STATUS_SUCCESS:
                first = 0;
                str += plen;
                len -= plen;
                if (len == 0) {
                    retval = __OCKL_AS_STATUS_SUCCESS;
                    thread_done = 1;
                }
                break;
            default:
                retval = status;
                thread_done = 1;
                break;
            }
        }
    } while (!__ockl_wfall_i32(thread_done));

    return retval;
}
