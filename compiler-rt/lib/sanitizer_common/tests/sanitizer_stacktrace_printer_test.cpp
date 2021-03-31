//===-- sanitizer_common_printer_test.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of sanitizer_common test suite.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_stacktrace_printer.h"

#include "gtest/gtest.h"

namespace __sanitizer {

TEST(SanitizerStacktracePrinter, RenderSourceLocation) {
  InternalScopedString str;
  RenderSourceLocation(&str, "/dir/file.cc", 10, 5, false, "");
  EXPECT_STREQ("/dir/file.cc:10:5", str.data());

  str.clear();
  RenderSourceLocation(&str, "/dir/file.cc", 11, 0, false, "");
  EXPECT_STREQ("/dir/file.cc:11", str.data());

  str.clear();
  RenderSourceLocation(&str, "/dir/file.cc", 0, 0, false, "");
  EXPECT_STREQ("/dir/file.cc", str.data());

  str.clear();
  RenderSourceLocation(&str, "/dir/file.cc", 10, 5, false, "/dir/");
  EXPECT_STREQ("file.cc:10:5", str.data());

  str.clear();
  RenderSourceLocation(&str, "/dir/file.cc", 10, 5, true, "");
  EXPECT_STREQ("/dir/file.cc(10,5)", str.data());

  str.clear();
  RenderSourceLocation(&str, "/dir/file.cc", 11, 0, true, "");
  EXPECT_STREQ("/dir/file.cc(11)", str.data());

  str.clear();
  RenderSourceLocation(&str, "/dir/file.cc", 0, 0, true, "");
  EXPECT_STREQ("/dir/file.cc", str.data());

  str.clear();
  RenderSourceLocation(&str, "/dir/file.cc", 10, 5, true, "/dir/");
  EXPECT_STREQ("file.cc(10,5)", str.data());
}

TEST(SanitizerStacktracePrinter, RenderModuleLocation) {
  InternalScopedString str;
  RenderModuleLocation(&str, "/dir/exe", 0x123, kModuleArchUnknown, "");
  EXPECT_STREQ("(/dir/exe+0x123)", str.data());

  // Check that we strip file prefix if necessary.
  str.clear();
  RenderModuleLocation(&str, "/dir/exe", 0x123, kModuleArchUnknown, "/dir/");
  EXPECT_STREQ("(exe+0x123)", str.data());

  // Check that we render the arch.
  str.clear();
  RenderModuleLocation(&str, "/dir/exe", 0x123, kModuleArchX86_64H, "/dir/");
  EXPECT_STREQ("(exe:x86_64h+0x123)", str.data());
}

TEST(SanitizerStacktracePrinter, RenderFrame) {
  int frame_no = 42;
  AddressInfo info;
  info.address = 0x400000;
  info.module = internal_strdup("/path/to/my/module");
  info.module_offset = 0x200;
  info.function = internal_strdup("function_foo");
  info.function_offset = 0x100;
  info.file = internal_strdup("/path/to/my/source");
  info.line = 10;
  info.column = 5;
  InternalScopedString str;

  // Dump all the AddressInfo fields.
  RenderFrame(&str,
              "%% Frame:%n PC:%p Module:%m ModuleOffset:%o "
              "Function:%f FunctionOffset:%q Source:%s Line:%l "
              "Column:%c",
              frame_no, info.address, &info, false, "/path/to/", "function_");
  EXPECT_STREQ("% Frame:42 PC:0x400000 Module:my/module ModuleOffset:0x200 "
               "Function:foo FunctionOffset:0x100 Source:my/source Line:10 "
               "Column:5",
               str.data());
  info.Clear();
  str.clear();

  // Test special format specifiers.
  info.address = 0x400000;
  RenderFrame(&str, "%M", frame_no, info.address, &info, false);
  EXPECT_NE(nullptr, internal_strstr(str.data(), "400000"));
  str.clear();

  RenderFrame(&str, "%L", frame_no, info.address, &info, false);
  EXPECT_STREQ("(<unknown module>)", str.data());
  str.clear();

  info.module = internal_strdup("/path/to/module");
  info.module_offset = 0x200;
  RenderFrame(&str, "%M", frame_no, info.address, &info, false);
  EXPECT_NE(nullptr, internal_strstr(str.data(), "(module+0x"));
  EXPECT_NE(nullptr, internal_strstr(str.data(), "200"));
  str.clear();

  RenderFrame(&str, "%L", frame_no, info.address, &info, false);
  EXPECT_STREQ("(/path/to/module+0x200)", str.data());
  str.clear();

  info.function = internal_strdup("my_function");
  RenderFrame(&str, "%F", frame_no, info.address, &info, false);
  EXPECT_STREQ("in my_function", str.data());
  str.clear();

  info.function_offset = 0x100;
  RenderFrame(&str, "%F %S", frame_no, info.address, &info, false);
  EXPECT_STREQ("in my_function+0x100 <null>", str.data());
  str.clear();

  info.file = internal_strdup("my_file");
  RenderFrame(&str, "%F %S", frame_no, info.address, &info, false);
  EXPECT_STREQ("in my_function my_file", str.data());
  str.clear();

  info.line = 10;
  RenderFrame(&str, "%F %S", frame_no, info.address, &info, false);
  EXPECT_STREQ("in my_function my_file:10", str.data());
  str.clear();

  info.column = 5;
  RenderFrame(&str, "%S %L", frame_no, info.address, &info, false);
  EXPECT_STREQ("my_file:10:5 my_file:10:5", str.data());
  str.clear();

  RenderFrame(&str, "%S %L", frame_no, info.address, &info, true);
  EXPECT_STREQ("my_file(10,5) my_file(10,5)", str.data());
  str.clear();

  info.column = 0;
  RenderFrame(&str, "%F %S", frame_no, info.address, &info, true);
  EXPECT_STREQ("in my_function my_file(10)", str.data());
  str.clear();

  info.line = 0;
  RenderFrame(&str, "%F %S", frame_no, info.address, &info, true);
  EXPECT_STREQ("in my_function my_file", str.data());
  str.clear();

  info.Clear();
}

}  // namespace __sanitizer
