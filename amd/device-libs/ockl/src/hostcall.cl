/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"

/** \brief Internal implementation of hostcall.
 *
 *  *** INTERNAL USE ONLY ***
 *  Internal function, not safe for direct use in user
 *  code. Application kernels must only use __ockl_hostcall_preview()
 *  defined below.
 */
__attribute__((cold))
extern long2
__ockl_hostcall_internal(void *buffer, uint service_id,
                         ulong arg0, ulong arg1, ulong arg2, ulong arg3,
                         ulong arg4, ulong arg5, ulong arg6, ulong arg7);

/** \brief Submit a wave-wide hostcall packet.
 *  \param service_id The service to be invoked on the host.
 *  \param arg0 Up to eight parameters (arg0..arg7)
 *  \return Two 64-bit values.
 *
 *  The hostcall is executed for all active threads in the
 *  wave. #service_id must be uniform across the active threads,
 *  otherwise behaviour is undefined. The service parameters may be
 *  different for each active thread, and correspondingly, the
 *  returned values are also different.
 *
 *  The contents of the input parameters and the return values are
 *  defined by the service being invoked.
 *
 *  *** PREVIEW FEATURE ***
 *  This is a feature preview and considered alpha quality only;
 *  behaviour may vary between ROCm releases. Device code that invokes
 *  hostcall can be launched only on the ROCm release that it was
 *  compiled for, otherwise behaviour is undefined.
 */
long2
__ockl_hostcall_preview(uint service_id,
                        ulong arg0, ulong arg1, ulong arg2, ulong arg3,
                        ulong arg4, ulong arg5, ulong arg6, ulong arg7)
{
    void *buffer;
    if (__oclc_ABI_version < 500) {
        buffer = (__global void *)((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[3];
    } else {
        buffer = (__global void *)((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[10];
    }

    return __ockl_hostcall_internal(buffer, service_id, arg0, arg1, arg2, arg3,
                                    arg4, arg5, arg6, arg7);
}
