//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for net/if.h and linux/if.h compatibility.
///
//===----------------------------------------------------------------------===//

// Include our header first.
#include <net/if.h>

// And Linux header afterwards. The blank line prevents clang-format from
// reordering these.
#include <linux/if.h>

#include "test/UnitTest/Test.h"

TEST(LlvmLibcNetIfAndLinuxIfTest, LinuxIfAfterNetIf) {
  // Test that <linux/if.h> can be included after <net/if.h>.
  struct ifreq ifr;
  (void)ifr;
}
