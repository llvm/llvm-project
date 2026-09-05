//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for which run path a module offers
///
//===----------------------------------------------------------------------===//

#include "src/__support/elf/search.h"
#include "test/UnitTest/Test.h"

#include <link.h>

using LIBC_NAMESPACE::elf::Module;
using LIBC_NAMESPACE::elf::rpath_of;
using LIBC_NAMESPACE::elf::runpath_of;

namespace {

const char STRTAB[] = "\0/where/it/will/live\0/where/it/is/now";
constexpr ElfW(Word) LIVE = 1;
constexpr ElfW(Word) NOW = 21;

ElfW(Dyn) DYNAMIC[4];
ElfW(Phdr) PHDRS[1];

// A module carrying whichever of the two tags the caller asks for.
Module with(bool rpath, bool runpath) {
  size_t n = 0;
  if (rpath)
    DYNAMIC[n++] = {DT_RPATH, {NOW}};
  if (runpath)
    DYNAMIC[n++] = {DT_RUNPATH, {LIVE}};
  DYNAMIC[n++] = {DT_STRTAB, {reinterpret_cast<ElfW(Addr)>(STRTAB)}};
  DYNAMIC[n++] = {DT_NULL, {0}};

  PHDRS[0].p_type = PT_DYNAMIC;
  PHDRS[0].p_vaddr = reinterpret_cast<ElfW(Addr)>(DYNAMIC);
  PHDRS[0].p_memsz = n * sizeof(ElfW(Dyn));
  return Module(PHDRS, 1, /*load_bias=*/0, "libtest.so.1");
}

} // anonymous namespace

TEST(LlvmLibcElfSearchPathsTest, NeitherTagOffersNothing) {
  Module m = with(/*rpath=*/false, /*runpath=*/false);
  ASSERT_TRUE(rpath_of(m) == nullptr);
  ASSERT_TRUE(runpath_of(m) == nullptr);
}

TEST(LlvmLibcElfSearchPathsTest, TheOlderTagOnItsOwn) {
  Module m = with(/*rpath=*/true, /*runpath=*/false);
  ASSERT_STREQ(rpath_of(m), STRTAB + NOW);
  ASSERT_TRUE(runpath_of(m) == nullptr);
}

TEST(LlvmLibcElfSearchPathsTest, TheNewerTagOnItsOwn) {
  Module m = with(/*rpath=*/false, /*runpath=*/true);
  ASSERT_TRUE(rpath_of(m) == nullptr);
  ASSERT_STREQ(runpath_of(m), STRTAB + LIVE);
}

// A module carrying both is not using the older one at all. Only the newer
// one is offered, and it is looked at after LD_LIBRARY_PATH rather than
// before it, so what the two are called apart decides whether the caller can
// override the path.
TEST(LlvmLibcElfSearchPathsTest, TheNewerTagTurnsTheOlderOneOff) {
  Module m = with(/*rpath=*/true, /*runpath=*/true);
  ASSERT_TRUE(rpath_of(m) == nullptr);
  ASSERT_STREQ(runpath_of(m), STRTAB + LIVE);
}
