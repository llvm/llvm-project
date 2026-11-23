//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Loader test to check if r_debug is set correctly.
///
//===----------------------------------------------------------------------===//

#include "src/link/_r_debug.h"
#include "test/IntegrationTest/test.h"

extern "C" [[gnu::weak]] ElfW(Dyn) _DYNAMIC[];

TEST_MAIN() {
  ASSERT_EQ(LIBC_NAMESPACE::_r_debug.r_version, 1);
  ASSERT_EQ(LIBC_NAMESPACE::_r_debug.r_state, static_cast<int>(RT_CONSISTENT));
  ASSERT_NE(LIBC_NAMESPACE::_r_debug.r_map,
            static_cast<struct link_map *>(nullptr));
  ASSERT_NE(LIBC_NAMESPACE::_r_debug.r_brk, static_cast<ElfW(Addr)>(0));
  ASSERT_EQ(LIBC_NAMESPACE::_r_debug.r_ldbase,
            LIBC_NAMESPACE::_r_debug.r_map->l_addr);

  struct link_map *map = LIBC_NAMESPACE::_r_debug.r_map;
  ASSERT_STREQ(map->l_name, "");
  ASSERT_EQ(map->l_ld, _DYNAMIC);
  ASSERT_EQ(map->l_next, static_cast<struct link_map *>(nullptr));
  ASSERT_EQ(map->l_prev, static_cast<struct link_map *>(nullptr));

  return 0;
}
