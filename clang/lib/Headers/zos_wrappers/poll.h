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
#if __has_include_next(<poll.h>)
#include_next <poll.h>
#ifdef __poll
#undef __poll
#define __poll __poll
#endif
#endif /* __has_include_next(<poll.h>) */
#endif /* __ZOS_WRAPPERS_POLL_H */
