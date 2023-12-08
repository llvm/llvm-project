// RUN: rm -rf %t
// RUN: %clang_cc1 -I%S/Inputs/overloadable-attrs -fmodules \
// RUN:            -fmodule-map-file=%S/Inputs/overloadable-attrs/module.modulemap \
// RUN:            -fmodules-cache-path=%t -verify %s -std=c++11
//
// Ensures that we don't merge decls with attrs that we allow overloading on.
//
// expected-no-diagnostics

#include "a.h"

static_assert(enable_if_attrs::fn1() == 1, "");
static_assert(enable_if_attrs::fn2() == 1, "");
static_assert(enable_if_attrs::fn3(0) == 0, "");
static_assert(enable_if_attrs::fn3(1) == 1, "");
static_assert(enable_if_attrs::fn4(0) == 0, "");
static_assert(enable_if_attrs::fn4(1) == 1, "");
static_assert(enable_if_attrs::fn5(0) == 0, "");
static_assert(enable_if_attrs::fn5(1) == 1, "");

static_assert(pass_object_size_attrs::fn1(nullptr) == 1, "");
static_assert(pass_object_size_attrs::fn2(nullptr) == 1, "");
