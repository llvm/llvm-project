/*===---- lifetimebound.h - Lifetime attributes -----------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===------------------------------------------------------------------------===
 */

#ifndef __LIFETIMEBOUND_H
#define __LIFETIMEBOUND_H

#define __lifetimebound __attribute__((lifetimebound))

#define __lifetime_capture_by(X) __attribute__((lifetime_capture_by(X)))

#define __noescape __attribute__((noescape))

#endif /* __LIFETIMEBOUND_H */
