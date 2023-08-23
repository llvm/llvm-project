/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#ifndef OCKL_PRIV_H
#define OCKL_PRIV_H

#define REQUIRES_WAVE32 __attribute__((target("wavefrontsize32")))
#define REQUIRES_WAVE64 __attribute__((target("wavefrontsize64")))

#endif // OCKL_PRIV_H
