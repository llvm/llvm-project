/*===----------------------------- poll.h ----------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __ZOS_WRAPPERS_POLL_H
#define __ZOS_WRAPPERS_POLL_H
#if defined(__MVS__)
#include_next <poll.h>
#ifdef __poll
#undef __poll
#define __poll __poll
#endif
#endif /* defined(__MVS__) */
#endif /* __ZOS_WRAPPERS_POLL_H */
