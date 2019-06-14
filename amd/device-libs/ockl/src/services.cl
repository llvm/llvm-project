/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

typedef enum {
    __OCKL_HOSTCALL_SERVICE_DEFAULT,
    __OCKL_HOSTCALL_SERVICE_FUNCTION_CALL
} __ockl_hostcall_service_id;

extern long2
__ockl_hostcall_preview(uint service_id,
                        ulong arg0, ulong arg1, ulong arg2, ulong arg3,
                        ulong arg4, ulong arg5, ulong arg6, ulong arg7);

long2
__ockl_call_host_function(ulong fptr,
                          ulong arg0, ulong arg1, ulong arg2, ulong arg3,
                          ulong arg4, ulong arg5, ulong arg6)
{
    return __ockl_hostcall_preview(__OCKL_HOSTCALL_SERVICE_FUNCTION_CALL,
                                   fptr, arg0, arg1, arg2, arg3,
                                   arg4, arg5, arg6);
}
