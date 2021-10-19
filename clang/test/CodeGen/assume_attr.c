// RUN: %clang_cc1 -emit-llvm -triple i386-linux-gnu %s -o - | FileCheck %s
// RUN: %clang_cc1 -x c -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t %s -emit-llvm -o - | FileCheck %s

// TODO: for "foo" and "bar", "after" is not added as it appears "after" the first use or definition respectively. There might be a way to allow that.

// CHECK:   define{{.*}} void @bar() #0
// CHECK:   define{{.*}} void @baz() #1
// CHECK:   declare{{.*}} void @foo() #2
// CHECK:      attributes #0
// CHECK-SAME:   "llvm.assume"="bar:before1,bar:before2,bar:before3,bar:def1,bar:def2"
// CHECK:      attributes #1
// CHECK-SAME:   "llvm.assume"="baz:before1,baz:before2,baz:before3,baz:def1,baz:def2,baz:after"
// CHECK:      attributes #2
// CHECK-SAME:   "llvm.assume"="foo:before1,foo:before2,foo:before3"

#ifndef HEADER
#define HEADER

/// foo: declarations only

__attribute__((assume("foo:before1"))) void foo(void);

__attribute__((assume("foo:before2")))
__attribute__((assume("foo:before3"))) void
foo(void);

/// baz: static function declarations and a definition

__attribute__((assume("baz:before1"))) static void baz(void);

__attribute__((assume("baz:before2")))
__attribute__((assume("baz:before3"))) static void
baz(void);

// Definition
__attribute__((assume("baz:def1,baz:def2"))) static void baz(void) { foo(); }

__attribute__((assume("baz:after"))) static void baz(void);

/// bar: external function declarations and a definition

__attribute__((assume("bar:before1"))) void bar(void);

__attribute__((assume("bar:before2")))
__attribute__((assume("bar:before3"))) void
bar(void);

// Definition
__attribute__((assume("bar:def1,bar:def2"))) void bar(void) { baz(); }

__attribute__((assume("bar:after"))) void bar(void);

/// back to foo

__attribute__((assume("foo:after"))) void foo(void);

#endif
