// RUN: %clang_cc1 -E -P %s | FileCheck %s

#line 10 "not_a_\\tab"
const char *escaped_backslash = __FILE__;
// CHECK: const char *escaped_backslash = "not_a_\\tab";

#line 20 "c:\moo\zar\haz.h"
const char *windows_path = __FILE__;
// CHECK: const char *windows_path = "c:\\moo\\zar\\haz.h";

#line 30 "original\x12source.c"
const char *non_escape = __FILE__;
// CHECK: const char *non_escape = "original\\x12source.c";

# 40 "gnu_\\path"
const char *gnu_line_marker = __FILE__;
// CHECK: const char *gnu_line_marker = "gnu_\\path";
