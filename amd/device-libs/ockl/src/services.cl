/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

typedef enum {
    SERVICE_DEFAULT,
    SERVICE_FUNCTION_CALL,
    SERVICE_PRINTF,
} service_id_t;

extern long2
__ockl_hostcall_preview(uint service_id, ulong arg0, ulong arg1, ulong arg2,
                        ulong arg3, ulong arg4, ulong arg5, ulong arg6,
                        ulong arg7);

/*===---  FUNCTION CALL  -----------------------------------------------------*/

long2
__ockl_call_host_function(ulong fptr, ulong arg0, ulong arg1, ulong arg2,
                          ulong arg3, ulong arg4, ulong arg5, ulong arg6)
{
    return __ockl_hostcall_preview(SERVICE_FUNCTION_CALL, fptr, arg0, arg1,
                                   arg2, arg3, arg4, arg5, arg6);
}

/*===---  MESSAGES  ----------------------------------------------------------*/

/** \brief Concatenating hostcalls into a message
 *
 *  A message is a stream of 64-bit integers transmitted as a series
 *  of hostcall invocations by the device code. Although the hostcall
 *  is "warp-wide", the message for each workitem is distinct.
 *
 *  Of the eight uint64_t arguments in hostcall, the first argument is
 *  used as the message descriptor, while the rest are used for
 *  message contents. The descriptor consists of the following fields:
 *
 *  - Bit  0     is the BEGIN flag.
 *  - Bit  1     is the END flag.
 *  - Bits 2-4   are reserved and must be zero.
 *  - Bits 5-7   indicate the number of elements being transmitted.
 *  - Bits 8-63  contain a 56-bit message ID.
 *
 *  A hostcall with the BEGIN flag set in the descriptor indicates the
 *  start of a new message. A hostcall with the END flag set indicates
 *  the end of a message. A single hostcall can have both flags set if
 *  the message fits in the payload of a single hostcall.  Each
 *  hostcall indicates the number of uint64_t elements in the payload
 *  that contain data to be appended to the message.
 *
 *  When the accumulator receives a hostcall with the BEGIN flag set,
 *  it allocates a new message ID, which is transmitted to the device
 *  via the first return value in the hostcall. Every subsequent
 *  hostcall containing the same message ID appends its payload to
 *  that message. The message is said to be "active" until a
 *  corresponding END hostcall is received.
 *
 *  When the accumulator receives a hostcall with the END flag set, it
 *  invokes the corresponding message handler on the contents of the
 *  accumulated message, and then discards the message. The handler
 *  may return up to two uint64_t values, that are transmitted to the
 *  device via the return value of the last hostcall.
 *
 *  Behaviour is undefined in each of the following cases:
 *  - An END packet is received with a non-existent message ID, or with
 *    the ID of a message that has previously been END'ed.
 *  - No END packet is received for an active message.
 *  - Any of the reserved bits are non-zero.
 *  - Different hostcalls indicate the same active message ID but a
 *    different service.
 */

/** Enums that describe the message descriptor fields.
 */
typedef enum {
    DESCRIPTOR_OFFSET_FLAG_BEGIN = 0,
    DESCRIPTOR_OFFSET_FLAG_END = 1,
    DESCRIPTOR_OFFSET_RESERVED0 = 2,
    DESCRIPTOR_OFFSET_LEN = 5,
    DESCRIPTOR_OFFSET_ID = 8
} descriptor_offset_t;

typedef enum {
    DESCRIPTOR_WIDTH_FLAG_BEGIN = 1,
    DESCRIPTOR_WIDTH_FLAG_END = 1,
    DESCRIPTOR_WIDTH_RESERVED0 = 3,
    DESCRIPTOR_WIDTH_LEN = 3,
    DESCRIPTOR_WIDTH_ID = 56
} descriptor_width_t;

static ulong
msg_set_len(ulong pd, uint len)
{
    ulong reset_mask =
        ~(((1UL << DESCRIPTOR_WIDTH_LEN) - 1) << DESCRIPTOR_OFFSET_LEN);
    return (pd & reset_mask) | ((ulong)len << DESCRIPTOR_OFFSET_LEN);
}

static ulong
msg_set_begin_flag(ulong pd)
{
    return pd | (1UL << DESCRIPTOR_OFFSET_FLAG_BEGIN);
}

static ulong
msg_reset_begin_flag(ulong pd)
{
    return pd & (~(1UL << DESCRIPTOR_OFFSET_FLAG_BEGIN));
}

static ulong
msg_get_end_flag(ulong pd)
{
    return pd & (1UL << DESCRIPTOR_OFFSET_FLAG_END);
}

static ulong
msg_reset_end_flag(ulong pd)
{
    return pd & (~(1UL << DESCRIPTOR_OFFSET_FLAG_END));
}

static ulong
msg_set_end_flag(ulong pd)
{
    return pd | (1UL << DESCRIPTOR_OFFSET_FLAG_END);
}

static long2
append_bytes(uint service_id, ulong msg_desc, const uchar *data, uint len)
{
    msg_desc = msg_set_len(msg_desc, (len + 7) / 8);

#define PACK_ULONG(ARG)                                                        \
    ulong ARG = 0;                                                             \
    if (len >= 8) {                                                            \
        ARG = (ulong)data[0] | ((ulong)data[1] << 8) |                         \
              ((ulong)data[2] << 16) | ((ulong)data[3] << 24) |                \
              ((ulong)data[4] << 32) | ((ulong)data[5] << 40) |                \
              ((ulong)data[6] << 48) | ((ulong)data[7] << 56);                 \
        len -= 8;                                                              \
        data += 8;                                                             \
    } else {                                                                   \
        for (uint ii = 0; ii != len; ++ii) {                                   \
            ARG |= (ulong)data[ii] << (ii * 8);                                \
        }                                                                      \
        len = 0;                                                               \
    }

    PACK_ULONG(arg1);
    PACK_ULONG(arg2);
    PACK_ULONG(arg3);
    PACK_ULONG(arg4);
    PACK_ULONG(arg5);
    PACK_ULONG(arg6);
    PACK_ULONG(arg7);

    return __ockl_hostcall_preview(service_id, msg_desc, arg1, arg2, arg3, arg4,
                                   arg5, arg6, arg7);
}

/** \brief Append an array of bytes to a message.
 *  \param service_id Identifier for the target host-side service.
 *  \param msg_desc   Message descriptor for a new or existing message.
 *  \param data       Pointer to an array of bytes.
 *  \param len        Length of the array.
 *  \return Values depend on the state of the message.
 *
 *  The function can transmit a byte array of arbitrary length, but
 *  during transmission, the array is padded with zeroes until the
 *  length is a multiple of eight bytes. Only the array contents are
 *  transmitted, and not the length.
 *
 *  If the END flag is set, the function returns two long values
 *  received from the host message handler. Otherwise, the first
 *  return value is the message descriptor to be used for a subsequent
 *  message call, while the second return value is not defined.
 */
static long2
message_append_bytes(uint service_id, ulong msg_desc, const uchar *data,
                     ulong len)
{
    ulong end_flag = msg_get_end_flag(msg_desc);
    long2 retval = {0, 0};
    retval.x = msg_reset_end_flag(msg_desc);

    do {
        uint plen = len;
        if (len > 56) {
            plen = 56;
        } else {
            retval.x |= end_flag;
        }
        retval = append_bytes(service_id, retval.x, data, plen);
        len -= plen;
        data += plen;
    } while (len != 0);

    return retval;
}

/** \brief Append up to seven ulong values to a message.
 *  \param service_id Identifier for the target host-side service.
 *  \param msg_desc   Message descriptor for a new or existing message.
 *  \param num_args   Number of arguments to be appended (maximum seven).
 *  \param arg[0..6]  Arguments to be appended.
 *  \return Values depend on the state of the message.
 *
 *  Only the first #num_args arguments are appended to the
 *  message. The remaining arguments are ignored. Behaviour is
 *  undefined if #num_args is greater then seven.
 *
 *  If the END flag is set, the function returns two uint64_t values
 *  received from the host message handler. Otherwise, the first
 *  return value is the message descriptor to be used for a subsequent
 *  message call, while the second return value is not defined.
 */
static long2
message_append_args(uint service_id, ulong msg_desc, uint num_args, ulong arg0,
                    ulong arg1, ulong arg2, ulong arg3, ulong arg4, ulong arg5,
                    ulong arg6)
{
    msg_desc = msg_set_len(msg_desc, num_args);

    return __ockl_hostcall_preview(service_id, msg_desc, arg0, arg1, arg2, arg3,
                                   arg4, arg5, arg6);
}

/*===---  PRINTF  ------------------------------------------------------------*/

/** \brief Begin a new printf message.
 *  \param version          Must be zero.
 *  \return Message descriptor for a new printf invocation.
 */
ulong
__ockl_printf_begin(ulong version)
{
    const ulong msg_desc = msg_set_begin_flag(0);

    long2 retval = message_append_args(SERVICE_PRINTF, msg_desc, 1, version, 0,
                                       0, 0, 0, 0, 0);
    return retval.x;
}

/** \brief Append up to seven arguments to the printf message.
 *  \param msg_desc  Message descriptor for the current printf.
 *  \param num_args   Number of arguments to be appended (maximum seven).
 *  \param value0... The argument values to be appended.
 *  \param is_last   If non-zero, this causes the printf to be completed.
 *  \return Value depends on #is_last.
 *
 *  Only the first #num_args arguments are appended to the
 *  message. The remaining arguments are ignored. Behaviour is
 *  undefined if #num_args is greater then seven.
 *
 *  If #is_last is zero, the function returns a message desciptor that
 *  must be used by a subsequent call to any __ockl_printf*
 *  function. If #is_last is non-zero, the function causes the current
 *  printf to be completed on the host-side, and returns the value
 *  returned by that printf.
 */
ulong
__ockl_printf_append_args(ulong msg_desc, uint num_args, ulong value0,
                          ulong value1, ulong value2, ulong value3,
                          ulong value4, ulong value5, ulong value6,
                          uint is_last)
{
    if (is_last) {
        msg_desc = msg_set_end_flag(msg_desc);
    }

    long2 retval =
        message_append_args(SERVICE_PRINTF, msg_desc, num_args, value0, value1,
                            value2, value3, value4, value5, value6);
    return retval.x;
}

/** \brief Append a null-terminated string to the printf message.
 *  \param msg_desc Message descriptor for the current printf.
 *  \param data     Pointer to the string.
 *  \param length   Number of bytes, including the null terminator.
 *  \param is_last  If non-zero, this causes the printf to be completed.
 *  \return Value depends on #is_last.
 *
 *  The function appends a single null-terminated string to a current
 *  printf message, including the final null character. The host-side
 *  can use the bytes as a null-terminated string in place, without
 *  having to first copy the string and then append the null
 *  terminator.
 *
 *  #length itself is not transmitted. Behaviour is undefined if
 *  #length does not include the final null character. #data may
 *  be a null pointer, in which case, #length is ignored and a single
 *  zero is transmitted. This makes the nullptr indistinguishable from
 *  an empty string to the host-side receiver.
 *
 *  The call to message_append_args() ensures that during
 *  transmission, the string is null-padded to a multiple of eight.
 *
 *  If #is_last is zero, the function returns a message desciptor that
 *  must be used by a subsequent call to any __ockl_printf*
 *  function. If #is_last is non-zero, the function causes the current
 *  printf to be completed on the host-side, and returns the value
 *  returned by that printf.
 */
ulong
__ockl_printf_append_string_n(ulong msg_desc, const char *data, ulong length,
                              uint is_last)
{
    long2 retval = {0, 0};

    if (is_last) {
        msg_desc = msg_set_end_flag(msg_desc);
    }

    if (!data) {
        retval = message_append_args(SERVICE_PRINTF, msg_desc, 1, 0, 0, 0, 0, 0,
                                     0, 0);
        return retval.x;
    }

    retval = message_append_bytes(SERVICE_PRINTF, msg_desc, (const uchar *)data,
                                  length);
    return retval.x;
}
