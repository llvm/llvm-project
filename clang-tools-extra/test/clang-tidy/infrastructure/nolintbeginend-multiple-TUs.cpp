// RUN: clang-tidy %S/Inputs/nolintbeginend/1st-translation-unit.cpp %S/Inputs/nolintbeginend/2nd-translation-unit.cpp --checks='-*,google-explicit-constructor' 2>&1 | FileCheck %s

// CHECK-NOT: 1st-translation-unit.cpp:2:11: warning: single-argument constructors must be marked explicit
// CHECK: 1st-translation-unit.cpp:5:11: warning: single-argument constructors must be marked explicit
// CHECK: 2nd-translation-unit.cpp:2:11: warning: single-argument constructors must be marked explicit
// CHECK-NOT: 2nd-translation-unit.cpp:5:11: warning: single-argument constructors must be marked explicit
