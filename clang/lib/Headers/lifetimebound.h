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

#if defined(__cplusplus) && defined(__has_cpp_attribute)
#define __use_cpp_spelling(x) __has_cpp_attribute(x)
#else
#define __use_cpp_spelling(x) 0
#endif

#if __use_cpp_spelling(clang::lifetimebound)
#define __lifetimebound [[clang::lifetimebound]]
#else
#define __lifetimebound __attribute__((lifetimebound))
#endif

#if __use_cpp_spelling(clang::lifetime_capture_by)
#define __lifetime_capture_by(X) [[clang::lifetime_capture_by(X)]]
#else
#define __lifetime_capture_by(X) __attribute__((lifetime_capture_by(X)))
#endif

#if __use_cpp_spelling(clang::noescape)
#define __noescape [[clang::noescape]]
#else
#define __noescape __attribute__((noescape))
#endif

#endif /* __LIFETIMEBOUND_H */
