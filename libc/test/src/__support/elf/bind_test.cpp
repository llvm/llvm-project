//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for cross module symbol binding
///
//===----------------------------------------------------------------------===//

#include "src/__support/elf/bind.h"
#include "src/__support/elf/load_module.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::elf::bind_module;
using LIBC_NAMESPACE::elf::BindResult;
using LIBC_NAMESPACE::elf::load_module;
using LIBC_NAMESPACE::elf::LoadedModule;
using LIBC_NAMESPACE::elf::Module;
using LIBC_NAMESPACE::elf::SearchOrder;
using LIBC_NAMESPACE::elf::unmap_module;

namespace {
constexpr size_t PAGE = 4096;
constexpr const char *LIBC_PATH = LIBC_TEST_SHARED_OBJECT;
constexpr const char *LIBM_PATH = LIBM_TEST_SHARED_OBJECT;
} // anonymous namespace

TEST(LlvmLibcElfBindTest, ResolvesAcrossModules) {
  auto libc = load_module(LIBC_PATH, PAGE);
  ASSERT_TRUE(libc.has_value());
  LoadedModule c = libc.value();

  Module modules[] = {c.module};
  SearchOrder order(modules, 1);

  // Defined by libc.so, so the search order must find it at a real address.
  auto printf_addr = order.resolve("printf");
  ASSERT_TRUE(printf_addr.has_value());
  EXPECT_GT(*printf_addr, ElfW(Addr)(0));
  // Inside the mapping it was loaded into.
  EXPECT_GE(*printf_addr, ElfW(Addr)(c.mapping.load_bias));

  EXPECT_FALSE(order.resolve("definitely_not_a_libc_symbol").has_value());

  unmap_module(c.mapping);
}

TEST(LlvmLibcElfBindTest, BindsLibmAgainstLibc) {
  // libm.so records a DT_NEEDED on libc.so, so this is a real two module
  // case rather than a synthetic one.
  auto libc = load_module(LIBC_PATH, PAGE);
  ASSERT_TRUE(libc.has_value());
  auto libm = load_module(LIBM_PATH, PAGE);
  ASSERT_TRUE(libm.has_value());

  LoadedModule c = libc.value();
  LoadedModule m = libm.value();

  // libm names libc among its dependencies.
  bool needs_libc = false;
  m.module.for_each_needed([&needs_libc](const char *name) {
    const char *p = name;
    // Match the "libc.so" prefix without pulling in a string routine.
    const char *want = "libc.so";
    while (*want != '\0' && *p == *want) {
      ++p;
      ++want;
    }
    if (*want == '\0')
      needs_libc = true;
  });
  EXPECT_TRUE(needs_libc);

  // Search order is the target first, then its dependency, as a loader does.
  Module modules[] = {m.module, c.module};
  SearchOrder order(modules, 2);

  BindResult result = bind_module(m.module, 0, order);
  // Every strong symbol libm imports must have been found.
  EXPECT_EQ(result.unresolved, size_t(0));

  unmap_module(m.mapping);
  unmap_module(c.mapping);
}

TEST(LlvmLibcElfBindTest, EmptySearchOrderResolvesNothing) {
  SearchOrder order(nullptr, 0);
  EXPECT_FALSE(order.resolve("printf").has_value());
}
