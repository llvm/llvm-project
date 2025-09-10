/*===---- availability_domain.h - Availability Domain -----------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __AVAILABILITY_DOMAIN_H
#define __AVAILABILITY_DOMAIN_H

#include <stdint.h>

#define __AVAILABILITY_DOMAIN_ENABLED 0
#define __AVAILABILITY_DOMAIN_DISABLED 1
#define __AVAILABILITY_DOMAIN_DYNAMIC 2
#define __AVAILABILITY_DOMAIN_ALWAYS_ENABLED 3

/// Describes the fields of a Clang availability domain. This struct is an
/// implementation detail of the compiler and is subject to change so don't
/// reference `__AvailabilityDomain` directly. Instead, use the provided macros:
///
///     CLANG_DYNAMIC_AVAILABILITY_DOMAIN(MyDomain, query);
///
struct __AvailabilityDomain {
  /// The state of the domain (AVAILABLE, UNAVAILABLE, DYNAMIC, etc.).
  intptr_t state;
  /// An optional function pointer to call to query the availability of a domain
  /// at runtime. This should only be non-null for domains in the DYNAMIC state.
  int (*const runtimeQuery)(void);
};

#define CLANG_DYNAMIC_AVAILABILITY_DOMAIN(domain, query)                       \
  static struct __AvailabilityDomain domain __attribute__((                    \
      availability_domain(domain))) = {__AVAILABILITY_DOMAIN_DYNAMIC, query}

#define CLANG_ENABLED_AVAILABILITY_DOMAIN(domain)                              \
  static struct __AvailabilityDomain domain __attribute__((                    \
      availability_domain(domain))) = {__AVAILABILITY_DOMAIN_ENABLED, 0}

#define CLANG_DISABLED_AVAILABILITY_DOMAIN(domain)                             \
  static struct __AvailabilityDomain domain __attribute__((                    \
      availability_domain(domain))) = {__AVAILABILITY_DOMAIN_DISABLED, 0}

#define CLANG_ALWAYS_ENABLED_AVAILABILITY_DOMAIN(domain)                       \
  static struct __AvailabilityDomain domain                                    \
      __attribute__((availability_domain(domain))) = {                         \
          __AVAILABILITY_DOMAIN_ALWAYS_ENABLED, 0}

#endif /* __AVAILABILITY_DOMAIN_H */
