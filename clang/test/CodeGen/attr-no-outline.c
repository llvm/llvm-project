// RUN: %clang_cc1 -emit-llvm -x c %s -triple x86_64-unknown-linux-gnu -o - -femit-all-decls -fblocks | FileCheck %s --check-prefix=C
// RUN: %clang_cc1 -emit-llvm -x c++ %s -triple x86_64-unknown-linux-gnu -o - -femit-all-decls -fblocks | FileCheck %s --check-prefix=CXX
// RUN: %clang_cc1 -emit-llvm -x c++ %s -triple x86_64-unknown-linux-gnu -o - -femit-all-decls -fblocks -std=c++23 | FileCheck %s --check-prefixes=CXX,CXX23

// C-LABEL: define dso_local i32 @toplevel_func(
// C-SAME: ) #[[ATTR0:[0-9]+]] {

// CXX-LABEL: define dso_local noundef i32 @_Z13toplevel_funci(
// CXX-SAME: ) #[[ATTR0:[0-9]+]] {
[[clang::no_outline]] int toplevel_func(int x) {
  return x;
}


// C-only: Function without prototype
#ifndef __cplusplus
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-non-prototype"
#pragma clang diagnostic ignored "-Wimplicit-int"

// C-LABEL: define dso_local i32 @no_proto_func(
// C-SAME: ) #[[ATTR0]] {

[[clang::no_outline]] no_proto_func(x)
int x; {
  return x;
}

#pragma clang diagnostic pop
#endif

// With Blocks
#if __has_feature(blocks)

int func_with_block(int x) {
// C-LABEL: define internal i32 @__func_with_block_block_invoke(
// C-SAME: ) #[[ATTR0]] {

// CXX-LABEL: define internal noundef i32 @___Z15func_with_blocki_block_invoke(
// CXX-SAME: ) #[[ATTR1:[0-9]+]] {

  int (^block)(int) = ^ __attribute__((no_outline)) int (int y) { return y; };

  return block(x);
}
#endif

// C++-only: Member Functions, Lambdas, Templates
#ifdef __cplusplus

struct my_struct {

// CXX-LABEL: define linkonce_odr noundef i32 @_ZN9my_struct11member_funcEi(
// CXX-SAME: ) #[[ATTR0]] comdat
  [[clang::no_outline]] int member_func(int x) {
    return x;
  }

// CXX-LABEL: define linkonce_odr noundef i32 @_ZN9my_struct11static_funcEi(
// CXX-SAME: ) #[[ATTR0]] comdat
  [[clang::no_outline]] static int static_func(int x) {
    return x;
  }
};

template <typename T> struct templated_struct {
  [[clang::no_outline]] T member_func(T x) {
    return x;
  }

  [[clang::no_outline]] static T static_func(T x) {
    return x;
  }
};

// CXX-LABEL: define weak_odr noundef i32 @_ZN16templated_structIiE11member_funcEi(
// CXX-SAME: ) #[[ATTR0]] comdat
// CXX-LABEL: define weak_odr noundef i32 @_ZN16templated_structIiE11static_funcEi(
// CXX-SAME: ) #[[ATTR0]] comdat
template struct templated_struct<int>;


#if __cplusplus >= 202302L
int func_with_lambda(int x) {
  // CXX23-LABEL: define internal noundef i32 @"_ZZ16func_with_lambdaiENK3$_0clEv"(
  // CXX23-SAME: ) #[[ATTR0]]
  auto lambda = [x][[clang::no_outline]]() -> int {
    return x;
  };

  return lambda();
}
#endif
#endif


// C: attributes #[[ATTR0]] = {
// C-SAME: nooutline
// C-SAME: }

// CXX: attributes #[[ATTR0]] = {
// CXX-SAME: nooutline
// CXX-SAME: }

// CXX: attributes #[[ATTR1]] = {
// CXX-SAME: nooutline
// CXX-SAME: }
