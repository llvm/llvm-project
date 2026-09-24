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

// Test that <linux/if.h> can be included after <net/if.h>.  Include our header
// first.
#include <net/if.h>

// And Linux header afterwards. The blank line prevents clang-format from
// reordering these.
#include <linux/if.h>

#include <stddef.h>
#include "test/UnitTest/Test.h"

extern const size_t LINUX_IFMAP_SIZE;
extern const size_t LINUX_IFMAP_ALIGN;
extern const size_t LINUX_IFREQ_SIZE;
extern const size_t LINUX_IFREQ_ALIGN;

TEST(LlvmLibcNetIfAndLinuxIfTest, LinuxIfAfterNetIf) {
  // Verify that our struct definitions match the size and alignment of the
  // kernel UAPI definitions in <linux/if.h> (captured in linux_if_helper.cpp).
  EXPECT_EQ(sizeof(struct ifmap), LINUX_IFMAP_SIZE);
  EXPECT_EQ(alignof(struct ifmap), LINUX_IFMAP_ALIGN);
  EXPECT_EQ(sizeof(struct ifreq), LINUX_IFREQ_SIZE);
  EXPECT_EQ(alignof(struct ifreq), LINUX_IFREQ_ALIGN);
}
