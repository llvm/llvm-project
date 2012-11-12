// RUN: %clang_cc1 -std=c++11 -emit-llvm -fblocks -o - -triple x86_64-apple-darwin10 %s | FileCheck %s
// rdar://11343499

namespace N {
  typedef void (^BL)();
  int func(BL, BL, BL);

// CHECK: define internal void @_ZN1N8ArrBlockE_block_invoke(
// CHECK: define internal void @_ZN1N8ArrBlockE_block_invoke_2(
// CHECK: define internal void @_ZN1N8ArrBlockE_block_invoke_3
  BL ArrBlock [] = { ^{}, ^{}, ^{} };

// CHECK: define internal void @_ZN1N4ivalE_block_invoke_4(
// CHECK: define internal void @_ZN1N4ivalE_block_invoke_5(
// CHECK: define internal void @_ZN1N4ivalE_block_invoke_6(
  int ival = func(^{}, ^{}, ^{});

// CHECK: define internal void @_ZN1N9gvarlobalE_block_invoke_7(
  void (^gvarlobal)(void) = ^{};

  struct S {
    BL field = ^{};
  };

// CHECK: define internal void @_ZN1N3blfE_block_invoke_8(
  S blf;
};
