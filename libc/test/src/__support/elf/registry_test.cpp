//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for the loaded module registry
///
//===----------------------------------------------------------------------===//

#include "src/__support/elf/registry.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::elf::load_module;
using LIBC_NAMESPACE::elf::LoadedModule;
using LIBC_NAMESPACE::elf::Module;
using LIBC_NAMESPACE::elf::Registry;
using LIBC_NAMESPACE::elf::RegistryEntry;
using LIBC_NAMESPACE::elf::unmap_module;

namespace {
constexpr size_t PAGE = 4096;
constexpr const char *LIBC_PATH = LIBC_TEST_SHARED_OBJECT;
constexpr const char *LIBM_PATH = LIBM_TEST_SHARED_OBJECT;
} // anonymous namespace

TEST(LlvmLibcElfRegistryTest, StartsEmpty) {
  Registry registry;
  EXPECT_EQ(registry.count(), size_t(0));
  EXPECT_TRUE(registry.find("anything") == nullptr);
  EXPECT_TRUE(registry.at(0) == nullptr);
}

TEST(LlvmLibcElfRegistryTest, TracksLoadedObjects) {
  Registry registry;
  auto libc = load_module(LIBC_PATH, PAGE);
  ASSERT_TRUE(libc.has_value());
  auto libm = load_module(LIBM_PATH, PAGE);
  ASSERT_TRUE(libm.has_value());

  RegistryEntry *c = registry.add(libc.value());
  RegistryEntry *m = registry.add(libm.value());
  ASSERT_TRUE(c != nullptr);
  ASSERT_TRUE(m != nullptr);
  EXPECT_EQ(registry.count(), size_t(2));
  EXPECT_EQ(c->references, size_t(1));

  // Looking up by the name each was opened under finds the same entry back.
  EXPECT_EQ(registry.find(LIBC_PATH), c);
  EXPECT_EQ(registry.find(LIBM_PATH), m);
  EXPECT_TRUE(registry.find("/not/loaded.so") == nullptr);

  // The search order covers everything loaded, in load order.
  Module order[4];
  size_t written = registry.collect(order, 4);
  EXPECT_EQ(written, size_t(2));

  unmap_module(m->loaded.mapping);
  unmap_module(c->loaded.mapping);
}

TEST(LlvmLibcElfRegistryTest, RemovedEntriesLeaveTheSearchOrder) {
  Registry registry;
  auto libc = load_module(LIBC_PATH, PAGE);
  ASSERT_TRUE(libc.has_value());
  RegistryEntry *c = registry.add(libc.value());
  ASSERT_TRUE(c != nullptr);

  Module order[2];
  EXPECT_EQ(registry.collect(order, 2), size_t(1));

  unmap_module(c->loaded.mapping);
  registry.remove(c);
  EXPECT_EQ(registry.collect(order, 2), size_t(0));
  EXPECT_TRUE(registry.find(LIBC_PATH) == nullptr);
}

TEST(LlvmLibcElfRegistryTest, CollectRespectsCapacity) {
  Registry registry;
  auto libc = load_module(LIBC_PATH, PAGE);
  ASSERT_TRUE(libc.has_value());
  auto libm = load_module(LIBM_PATH, PAGE);
  ASSERT_TRUE(libm.has_value());
  RegistryEntry *c = registry.add(libc.value());
  RegistryEntry *m = registry.add(libm.value());

  Module order[1];
  EXPECT_EQ(registry.collect(order, 1), size_t(1));

  unmap_module(m->loaded.mapping);
  unmap_module(c->loaded.mapping);
}
