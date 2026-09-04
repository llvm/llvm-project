// RUN: %clang_cc1 -emit-llvm -x c %s -triple x86_64-unknown-linux-gnu -o - -femit-all-decls -fblocks -DTEST_ATTR  | FileCheck %s --check-prefix=C,C-ATTR
// RUN: %clang_cc1 -emit-llvm -x c %s -triple x86_64-unknown-linux-gnu -o - -femit-all-decls -fblocks -mno-outline | FileCheck %s --check-prefix=C,C-ARG
// RUN: %clang_cc1 -emit-llvm -x c %s -triple x86_64-unknown-linux-gnu -o - -femit-all-decls -fblocks              | FileCheck %s --check-prefix=C,C-NONE


// RUN: %clang_cc1 -emit-llvm -x c++ %s -triple x86_64-unknown-linux-gnu -o - -femit-all-decls -fblocks -std=c++23 -DTEST_ATTR  | FileCheck %s --check-prefixes=CXX,CXX-ATTR
// RUN: %clang_cc1 -emit-llvm -x c++ %s -triple x86_64-unknown-linux-gnu -o - -femit-all-decls -fblocks -std=c++23 -mno-outline | FileCheck %s --check-prefixes=CXX,CXX-ARG
// RUN: %clang_cc1 -emit-llvm -x c++ %s -triple x86_64-unknown-linux-gnu -o - -femit-all-decls -fblocks -std=c++23              | FileCheck %s --check-prefixes=CXX,CXX-NONE

// This test checks that:
// - [[clang::no_outline]] adds the nooutline IR attribute to specific definitions
// - `-mno-outline` adds the nooutline IR attribute to all definitions
// - Lack of either does not add nooutline IR attribute

#ifdef TEST_ATTR
#define ATTR [[clang::no_outline]]
#define ATTR_DUNDER __attribute__((no_outline))
#else
#define ATTR
#define ATTR_DUNDER
#endif

// C-LABEL: define dso_local i32 @toplevel_func(
// C-SAME: ) #[[ATTR1:[0-9]+]] {

// CXX-LABEL: define dso_local noundef i32 @_Z13toplevel_funci(
// CXX-SAME: ) #[[ATTR1:[0-9]+]] {
ATTR int toplevel_func(int x) {
  return x;
}

// C-LABEL: define dso_local i32 @toplevel_func_noattr(
// C-ATTR-SAME: ) #[[ATTR2:[0-9]+]] {
// C-ARG-SAME:  ) #[[ATTR1]] {
// C-NONE-SAME: ) #[[ATTR1]] {

// CXX-LABEL: define dso_local noundef i32 @_Z20toplevel_func_noattri(
// CXX-ATTR-SAME: ) #[[ATTR2:[0-9]+]] {
// CXX-ARG-SAME:  ) #[[ATTR1]] {
// CXX-NONE-SAME: ) #[[ATTR1]] {
int toplevel_func_noattr(int x) {
  return x;
}

// C-only: Function without prototype
#ifndef __cplusplus
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-non-prototype"
#pragma clang diagnostic ignored "-Wimplicit-int"

// C-LABEL: define dso_local i32 @no_proto_func(
// C-SAME: ) #[[ATTR1]] {

ATTR no_proto_func(x)
int x; {
  return x;
}

#pragma clang diagnostic pop
#endif

// With Blocks
#if __has_feature(blocks)

// C-LABEL: define dso_local i32 @func_with_block(
// C-ATTR-SAME: ) #[[ATTR2]] {
// C-ARG-SAME:  ) #[[ATTR1]] {
// C-NONE-SAME: ) #[[ATTR1]] {

// CXX-LABEL: define dso_local noundef i32 @_Z15func_with_blocki(
// CXX-ATTR-SAME: ) #[[ATTR2]] {
// CXX-ARG-SAME:  ) #[[ATTR1]] {
// CXX-NONE-SAME: ) #[[ATTR1]] {
int func_with_block(int x) {

// C-LABEL: define internal i32 @__func_with_block_block_invoke(
// C-SAME: ) #[[ATTR1]] {

// CXX-LABEL: define internal noundef i32 @___Z15func_with_blocki_block_invoke(
// CXX-ATTR-SAME: ) #[[ATTR3:[0-9]+]] {
// CXX-ARG-SAME:  ) #[[ATTR2:[0-9]+]] {
// CXX-NONE-SAME: ) #[[ATTR2:[0-9]+]] {
  int (^block)(int) = ^ ATTR_DUNDER int (int y) { return y; };

  return block(x);
}
#endif

// C++-only: Member Functions, Lambdas, Templates
#ifdef __cplusplus

struct my_struct {

// CXX-LABEL: define linkonce_odr noundef i32 @_ZN9my_struct11member_funcEi(
// CXX-SAME: ) #[[ATTR1]] comdat
  ATTR int member_func(int x) {
    return x;
  }

// CXX-LABEL: define linkonce_odr noundef i32 @_ZN9my_struct11static_funcEi(
// CXX-SAME: ) #[[ATTR1]] comdat
  ATTR static int static_func(int x) {
    return x;
  }
};

template <typename T> struct templated_struct {
  ATTR T member_func(T x) {
    return x;
  }

  ATTR static T static_func(T x) {
    return x;
  }
};

// CXX-LABEL: define weak_odr noundef i32 @_ZN16templated_structIiE11member_funcEi(
// CXX-SAME: ) #[[ATTR1]] comdat
// CXX-LABEL: define weak_odr noundef i32 @_ZN16templated_structIiE11static_funcEi(
// CXX-SAME: ) #[[ATTR1]] comdat
template struct templated_struct<int>;


// CXX-LABEL: define dso_local noundef i32 @_Z16func_with_lambdai(
// CXX-ATTR-SAME: ) #[[ATTR2]]
// CXX-ARG-SAME:  ) #[[ATTR1]]
// CXX-NONE-SAME: ) #[[ATTR1]]
int func_with_lambda(int x) {

// CXX-LABEL: define internal noundef i32 @"_ZZ16func_with_lambdaiENK3$_0clEv"(
// CXX-SAME: ) #[[ATTR1]]
  auto lambda = [x] ATTR () -> int {
    return x;
  };

  return lambda();
}
#endif


// C: attributes #[[ATTR1]] = {
// C-ATTR-SAME: nooutline
// C-ARG-SAME: nooutline
// C-NONE-NOT: nooutline
// C-SAME: }

// C-ATTR: attributes #[[ATTR2]] = {
// C-ATTR-NOT: nooutline
// C-ATTR-SAME: }

// CXX: attributes #[[ATTR1]] = {
// CXX-ATTR-SAME: nooutline
// CXX-ARG-SAME: nooutline
// CXX-NONE-NOT: nooutline
// CXX-SAME: }

// CXX: attributes #[[ATTR2]] = {
// CXX-ATTR-NOT: nooutline
// CXX-ARG-SAME: nooutline
// CXX-NONE-NOT: nooutline
// CXX-SAME: }

// CXX-ATTR: attributes #[[ATTR3]] = {
// CXX-ATTR-SAME: nooutline
// CXX-ATTR-SAME: }
