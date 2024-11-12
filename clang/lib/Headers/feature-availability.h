/*===---- feature_availability.h - Feature Availability --------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __FEATURE_AVAILABILITY_H
#define __FEATURE_AVAILABILITY_H

#include <stdint.h>

/// The possible availability domain states. These values are hardcoded in the
/// compiler and reproduced here for convenience when defining domains.

#define __AVAILABILITY_DOMAIN_ENABLED 0
#define __AVAILABILITY_DOMAIN_DISABLED 1
#define __AVAILABILITY_DOMAIN_DYNAMIC 2

/// A struct describing availability domain definitions. This struct definition
/// is just a convenience to ensure that a header defining an availability
/// domain can define it with the arguments that Clang expects at parse time.
struct __AvailabilityDomain {
  /// The state of the domain (AVAILABLE, UNAVAILABLE, DYNAMIC, etc.).
  intptr_t state;
  /// An optional function pointer to call to query the availability of a domain
  /// at runtime. This should only be non-null for domains in the DYNAMIC state.
  int (*const runtimeQuery)(void);
};

#endif /* __FEATURE_AVAILABILITY_H */
