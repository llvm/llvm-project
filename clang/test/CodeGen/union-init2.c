// RUN: %clang_cc1 -emit-llvm %s -o - -triple i686-pc-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -x c++ %s -emit-llvm -triple x86_64-linux-gnu -o - | FileCheck %s --check-prefixes=CHECK-CXX

// Make sure we generate something sane instead of a ptrtoint
// CHECK: @r, [4 x i8] zeroinitializer
union x {long long b;union x* a;} r = {.a = &r};


// CHECK: global { [3 x i8], [5 x i8] } zeroinitializer
union z {
  char a[3];
  long long b;
};
union z y = {};

// CHECK: @foo = {{.*}}global %union.Foo undef, align 1
// CHECK-CXX: @foo = {{.*}}global %union.Foo undef, align 1
union Foo {
  struct Empty {} val;
};
union Foo foo = {};
