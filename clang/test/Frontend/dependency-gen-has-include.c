// Basic test
// RUN: rm -rf %t.dir
// RUN: mkdir %t.dir
// RUN: mkdir %t.dir/a
// RUN: echo "#ifndef HEADER_A" > %t.dir/a/header.h
// RUN: echo "#define HEADER_A" >> %t.dir/a/header.h
// RUN: echo "#endif" >> %t.dir/a/header.h
// RUN: mkdir %t.dir/system
// RUN: echo "#define SYSTEM_HEADER" > %t.dir/system/system-header.h
// RUN: mkdir %t.dir/next-a
// RUN: echo "#if __has_include_next(<next-header.h>)" > %t.dir/next-a/next-header.h
// RUN: echo "#endif" >> %t.dir/next-a/next-header.h
// RUN: mkdir %t.dir/next-b
// RUN: echo "#define NEXT_HEADER" > %t.dir/next-b/next-header.h

// RUN: %clang -MD -MF %t.dir/file.deps %s -fsyntax-only -I %t.dir -isystem %t.dir/system -I %t.dir/next-a -I %t.dir/next-b
// RUN: FileCheck -input-file=%t.dir/file.deps %s
// CHECK: dependency-gen-has-include.o
// CHECK: dependency-gen-has-include.c
// CHECK: a{{[/\\]}}header.h
// CHECK-NOT: missing{{[/\\]}}file.h
// CHECK: system{{[/\\]}}system-header.h
// CHECK: next-a{{[/\\]}}next-header.h
// CHECK: next-b{{[/\\]}}next-header.h

// Verify that we ignore system headers in user-only headers mode.
// RUN: %clang -MMD -MF %t.dir/user-headers.deps %s -fsyntax-only -I %t.dir -isystem %t.dir/system -I %t.dir/next-a -I %t.dir/next-b
// RUN: FileCheck -input-file=%t.dir/user-headers.deps --check-prefix CHECK-USER-HEADER %s
// CHECK-USER-HEADER-NOT: system{{[/\\]}}system-header.h

#if __has_include("a/header.h")
#endif
#if __has_include("missing/file.h")
#endif
#if __has_include(<system-header.h>)
#endif

#include <next-header.h>
