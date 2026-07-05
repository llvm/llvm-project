// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -I%S/Inputs/PR27890 -verify %s
// RUN: %clang_cc1 -std=c++11 -fmodules -fmodule-map-file=%S/Inputs/PR27890/module.modulemap -fmodules-cache-path=%t -I%S/Inputs/PR27890 -verify %s

#include "a.h"
enum ActionType {};
opt<ActionType> a(values(""));

// expected-no-diagnostics