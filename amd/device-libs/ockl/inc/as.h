#ifndef __OCKL_AS_H__
#define __OCKL_AS_H__

#include "ockl.h"

typedef enum {
    __OCKL_AS_STATUS_SUCCESS,
    __OCKL_AS_STATUS_INVALID_REQUEST,
    __OCKL_AS_STATUS_OUT_OF_RESOURCES,
    __OCKL_AS_STATUS_BUSY,
    __OCKL_AS_STATUS_UNKNOWN_ERROR
} __ockl_as_status_t;

typedef enum {
    __OCKL_AS_CONNECTION_BEGIN = 1,
    __OCKL_AS_CONNECTION_END = 2,
    __OCKL_AS_CONNECTION_FLUSH = 4
} __ockl_as_flag_t;

typedef enum {
    __OCKL_AS_BUILTIN_SERVICE_PRINTF = 42
} __ockl_as_builtin_service_t;

typedef struct {
    // Opaque handle. The value 0 is reserved.
    ulong handle;
} __ockl_as_signal_t;

typedef struct __ockl_as_packet_t __ockl_as_packet_t;

typedef struct {
    ulong read_index;
    ulong write_index;
    __ockl_as_signal_t doorbell_signal;
    __global __ockl_as_packet_t *base_address;
    ulong size;
} __ockl_as_stream_t;

__ockl_as_status_t
__ockl_as_write_block(__global __ockl_as_stream_t *stream, uchar service_id,
                      ulong *connection_id, const uchar *str, uint len,
                      uchar flags);

#endif
