// RUN: rm -rf %t
//
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodules-cache-path=%t \
// RUN:     -I %S/Inputs/initializer_list \
// RUN:     -fmodule-map-file=%S/Inputs/initializer_list/direct.modulemap \
// RUN:     %s -verify -std=c++11
//
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodules-cache-path=%t \
// RUN:     -I %S/Inputs/initializer_list \
// RUN:     -fmodule-map-file=%S/Inputs/initializer_list/indirect.modulemap \
// RUN:     %s -verify -std=c++11 -DINCLUDE_DIRECT

// expected-no-diagnostics

#ifdef INCLUDE_DIRECT
#include "direct.h"
auto k = {1, 2, 3};
#endif

@import initializer_list;

auto v = {1, 2, 3};
int n = std::min({1, 2, 3});
