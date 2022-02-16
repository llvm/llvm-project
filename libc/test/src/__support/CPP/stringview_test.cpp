//===-- Unittests for StringView ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/StringView.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcStringViewTest, InitializeCheck) {
  __llvm_libc::cpp::StringView v;
  ASSERT_EQ(v.size(), size_t(0));
  ASSERT_TRUE(v.data() == nullptr);

  v = __llvm_libc::cpp::StringView("");
  ASSERT_EQ(v.size(), size_t(0));
  ASSERT_TRUE(v.data() == nullptr);

  v = __llvm_libc::cpp::StringView(nullptr);
  ASSERT_EQ(v.size(), size_t(0));
  ASSERT_TRUE(v.data() == nullptr);

  v = __llvm_libc::cpp::StringView(nullptr, 10);
  ASSERT_EQ(v.size(), size_t(0));
  ASSERT_TRUE(v.data() == nullptr);

  v = __llvm_libc::cpp::StringView("abc", 0);
  ASSERT_EQ(v.size(), size_t(0));
  ASSERT_TRUE(v.data() == nullptr);

  v = __llvm_libc::cpp::StringView("123456789");
  ASSERT_EQ(v.size(), size_t(9));
}

TEST(LlvmLibcStringViewTest, Equals) {
  __llvm_libc::cpp::StringView v("abc");
  ASSERT_TRUE(v.equals(__llvm_libc::cpp::StringView("abc")));
  ASSERT_FALSE(v.equals(__llvm_libc::cpp::StringView()));
  ASSERT_FALSE(v.equals(__llvm_libc::cpp::StringView("")));
  ASSERT_FALSE(v.equals(__llvm_libc::cpp::StringView("123")));
  ASSERT_FALSE(v.equals(__llvm_libc::cpp::StringView("abd")));
  ASSERT_FALSE(v.equals(__llvm_libc::cpp::StringView("aaa")));
  ASSERT_FALSE(v.equals(__llvm_libc::cpp::StringView("abcde")));
}

TEST(LlvmLibcStringViewTest, startsWith) {
  __llvm_libc::cpp::StringView v("abc");
  ASSERT_TRUE(v.starts_with(__llvm_libc::cpp::StringView("a")));
  ASSERT_TRUE(v.starts_with(__llvm_libc::cpp::StringView("ab")));
  ASSERT_TRUE(v.starts_with(__llvm_libc::cpp::StringView("abc")));
  ASSERT_TRUE(v.starts_with(__llvm_libc::cpp::StringView()));
  ASSERT_TRUE(v.starts_with(__llvm_libc::cpp::StringView("")));
  ASSERT_FALSE(v.starts_with(__llvm_libc::cpp::StringView("123")));
  ASSERT_FALSE(v.starts_with(__llvm_libc::cpp::StringView("abd")));
  ASSERT_FALSE(v.starts_with(__llvm_libc::cpp::StringView("aaa")));
  ASSERT_FALSE(v.starts_with(__llvm_libc::cpp::StringView("abcde")));
}

TEST(LlvmLibcStringViewTest, RemovePrefix) {
  __llvm_libc::cpp::StringView v("123456789");

  auto p = v.remove_prefix(0);
  ASSERT_EQ(p.size(), size_t(9));
  ASSERT_TRUE(p.equals(__llvm_libc::cpp::StringView("123456789")));

  p = v.remove_prefix(4);
  ASSERT_EQ(p.size(), size_t(5));
  ASSERT_TRUE(p.equals(__llvm_libc::cpp::StringView("56789")));

  p = v.remove_prefix(9);
  ASSERT_EQ(p.size(), size_t(0));
  ASSERT_TRUE(p.data() == nullptr);

  p = v.remove_prefix(10);
  ASSERT_EQ(p.size(), size_t(0));
  ASSERT_TRUE(p.data() == nullptr);
}

TEST(LlvmLibcStringViewTest, RemoveSuffix) {
  __llvm_libc::cpp::StringView v("123456789");

  auto p = v.remove_suffix(0);
  ASSERT_EQ(p.size(), size_t(9));
  ASSERT_TRUE(p.equals(__llvm_libc::cpp::StringView("123456789")));

  p = v.remove_suffix(4);
  ASSERT_EQ(p.size(), size_t(5));
  ASSERT_TRUE(p.equals(__llvm_libc::cpp::StringView("12345")));

  p = v.remove_suffix(9);
  ASSERT_EQ(p.size(), size_t(0));
  ASSERT_TRUE(p.data() == nullptr);

  p = v.remove_suffix(10);
  ASSERT_EQ(p.size(), size_t(0));
  ASSERT_TRUE(p.data() == nullptr);
}

TEST(LlvmLibcStringViewTest, TrimSingleChar) {
  __llvm_libc::cpp::StringView v("     123456789   ");
  auto t = v.trim(' ');
  ASSERT_EQ(t.size(), size_t(9));
  ASSERT_TRUE(t.equals(__llvm_libc::cpp::StringView("123456789")));

  v = __llvm_libc::cpp::StringView("====12345==");
  t = v.trim(' ');
  ASSERT_EQ(v.size(), size_t(11));
  ASSERT_TRUE(t.equals(__llvm_libc::cpp::StringView("====12345==")));

  t = v.trim('=');
  ASSERT_EQ(t.size(), size_t(5));
  ASSERT_TRUE(t.equals(__llvm_libc::cpp::StringView("12345")));

  v = __llvm_libc::cpp::StringView("12345===");
  t = v.trim('=');
  ASSERT_EQ(t.size(), size_t(5));
  ASSERT_TRUE(t.equals(__llvm_libc::cpp::StringView("12345")));

  v = __llvm_libc::cpp::StringView("===========12345");
  t = v.trim('=');
  ASSERT_EQ(t.size(), size_t(5));
  ASSERT_TRUE(t.equals(__llvm_libc::cpp::StringView("12345")));

  v = __llvm_libc::cpp::StringView("============");
  t = v.trim('=');
  ASSERT_EQ(t.size(), size_t(0));
  ASSERT_TRUE(t.data() == nullptr);

  v = __llvm_libc::cpp::StringView();
  t = v.trim(' ');
  ASSERT_EQ(t.size(), size_t(0));
  ASSERT_TRUE(t.data() == nullptr);

  v = __llvm_libc::cpp::StringView("");
  t = v.trim(' ');
  ASSERT_EQ(t.size(), size_t(0));
  ASSERT_TRUE(t.data() == nullptr);
}
