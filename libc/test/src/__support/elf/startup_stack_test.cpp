//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for the process startup stack
///
//===----------------------------------------------------------------------===//

#include "src/__support/elf/startup_stack.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::elf::executable_from;
using LIBC_NAMESPACE::elf::StartupStack;

namespace {

char ARG0[] = "prog";
char ARG1[] = "--flag";
char ENV0[] = "PATH=/bin";
char ENV1[] = "HOME=/root";

ElfW(Phdr) FAKE_PHDRS[3];

// A stack laid out exactly as the kernel builds one.
uintptr_t *build_stack(bool with_auxv, unsigned long phent) {
  static uintptr_t words[32];
  size_t i = 0;
  words[i++] = 2; // argc
  words[i++] = reinterpret_cast<uintptr_t>(ARG0);
  words[i++] = reinterpret_cast<uintptr_t>(ARG1);
  words[i++] = 0; // end of argv
  words[i++] = reinterpret_cast<uintptr_t>(ENV0);
  words[i++] = reinterpret_cast<uintptr_t>(ENV1);
  words[i++] = 0; // end of envp
  if (with_auxv) {
    words[i++] = AT_PHDR;
    words[i++] = reinterpret_cast<uintptr_t>(FAKE_PHDRS);
    words[i++] = AT_PHNUM;
    words[i++] = 3;
    words[i++] = AT_PHENT;
    words[i++] = phent;
    words[i++] = AT_ENTRY;
    words[i++] = 0x401000;
    words[i++] = AT_PAGESZ;
    words[i++] = 4096;
  }
  words[i++] = AT_NULL;
  words[i++] = 0;
  return words;
}

} // anonymous namespace

TEST(LlvmLibcElfStartupStackTest, ParsesArgvAndEnvp) {
  StartupStack stack =
      StartupStack::from(build_stack(true, sizeof(ElfW(Phdr))));
  EXPECT_EQ(stack.argc(), uintptr_t(2));
  ASSERT_TRUE(stack.argv() != nullptr);
  EXPECT_STREQ(stack.argv()[0], "prog");
  EXPECT_STREQ(stack.argv()[1], "--flag");
  EXPECT_TRUE(stack.argv()[2] == nullptr);

  ASSERT_TRUE(stack.envp() != nullptr);
  EXPECT_STREQ(stack.envp()[0], "PATH=/bin");
  EXPECT_STREQ(stack.envp()[1], "HOME=/root");
  EXPECT_TRUE(stack.envp()[2] == nullptr);
  EXPECT_EQ(stack.envc(), size_t(2));
}

TEST(LlvmLibcElfStartupStackTest, ReadsAuxiliaryVector) {
  StartupStack stack =
      StartupStack::from(build_stack(true, sizeof(ElfW(Phdr))));
  ASSERT_TRUE(stack.auxval(AT_PAGESZ).has_value());
  EXPECT_EQ(*stack.auxval(AT_PAGESZ), 4096UL);
  EXPECT_EQ(*stack.auxval(AT_PHNUM), 3UL);
  EXPECT_EQ(*stack.auxval(AT_ENTRY), 0x401000UL);
  // Not supplied, so absent rather than zero.
  EXPECT_FALSE(stack.auxval(AT_BASE).has_value());
}

TEST(LlvmLibcElfStartupStackTest, EmptyAuxiliaryVector) {
  StartupStack stack =
      StartupStack::from(build_stack(false, sizeof(ElfW(Phdr))));
  EXPECT_EQ(stack.argc(), uintptr_t(2));
  EXPECT_FALSE(stack.auxval(AT_PHDR).has_value());
  EXPECT_FALSE(executable_from(stack).has_value());
}

TEST(LlvmLibcElfStartupStackTest, DescribesTheExecutable) {
  StartupStack stack =
      StartupStack::from(build_stack(true, sizeof(ElfW(Phdr))));
  auto image = executable_from(stack);
  ASSERT_TRUE(image.has_value());
  EXPECT_EQ(image->phdrs, static_cast<const ElfW(Phdr) *>(FAKE_PHDRS));
  EXPECT_EQ(image->phnum, ElfW(Half)(3));
  EXPECT_EQ(image->entry, ElfW(Addr)(0x401000));
  // AT_BASE absent means the program was not started through an interpreter.
  EXPECT_EQ(image->interpreter_base, ElfW(Addr)(0));
}

TEST(LlvmLibcElfStartupStackTest, RejectsMismatchedPhdrSize) {
  // A vector describing headers of a different size must be refused rather
  // than walked, since each step would read past the end of an entry.
  StartupStack stack =
      StartupStack::from(build_stack(true, sizeof(ElfW(Phdr)) + 8));
  EXPECT_FALSE(executable_from(stack).has_value());
}
