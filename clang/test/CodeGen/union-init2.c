// RUN: %clang_cc1 -emit-llvm %s -o - -triple i686-pc-linux-gnu | FileCheck %s

// Make sure we generate something sane instead of a ptrtoint
// CHECK: @r, [4 x i8] undef
union x {long long b;union x* a;} r = {.a = &r};


// CHECK: global { [3 x i8], [5 x i8] } { [3 x i8] zeroinitializer, [5 x i8] undef }
union z {
  char a[3];
  long long b;
};
union z y = {};
