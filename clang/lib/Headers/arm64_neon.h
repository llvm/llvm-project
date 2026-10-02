/*===---- arm64_neon.h - ARM64 NEON intrinsics -----------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

/* Only include this if we're compiling for the windows platform. */
#ifndef _MSC_VER
#include_next <arm64_neon.h>
#else

#ifndef __ARM64_NEON_H
#define __ARM64_NEON_H

#include <arm_neon.h>

#endif /* __ARM64_NEON_H */
#endif /* _MSC_VER */
