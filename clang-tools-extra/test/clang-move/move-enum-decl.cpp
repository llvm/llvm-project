// RUN: mkdir -p %t.dir/move-enum
// RUN: cp %S/Inputs/enum.h  %t.dir/move-enum/enum.h
// RUN: echo '#include "enum.h"' > %t.dir/move-enum/enum.cpp
// RUN: cd %t.dir/move-enum
//
// -----------------------------------------------------------------------------
// Test moving enum declarations.
// -----------------------------------------------------------------------------
// RUN: clang-move -names="a::E1" -new_cc=%t.dir/move-enum/new_test.cpp -new_header=%t.dir/move-enum/new_test.h -old_cc=%t.dir/move-enum/enum.cpp -old_header=%t.dir/move-enum/enum.h %t.dir/move-enum/enum.cpp -- -std=c++11
// RUN: FileCheck -input-file=%t.dir/move-enum/new_test.h -check-prefix=CHECK-NEW-TEST-H-CASE1 %s
// RUN: FileCheck -input-file=%t.dir/move-enum/enum.h -check-prefix=CHECK-OLD-TEST-H-CASE1 %s
//
// CHECK-NEW-TEST-H-CASE1: namespace a {
// CHECK-NEW-TEST-H-CASE1-NEXT: enum E1 { Green, Red };
// CHECK-NEW-TEST-H-CASE1-NEXT: }

// CHECK-OLD-TEST-H-CASE1-NOT: enum E1 { Green, Red };


// -----------------------------------------------------------------------------
// Test moving scoped enum declarations.
// -----------------------------------------------------------------------------
// RUN: cp %S/Inputs/enum.h  %t.dir/move-enum/enum.h
// RUN: echo '#include "enum.h"' > %t.dir/move-enum/enum.cpp
// RUN: clang-move -names="a::E2" -new_cc=%t.dir/move-enum/new_test.cpp -new_header=%t.dir/move-enum/new_test.h -old_cc=%t.dir/move-enum/enum.cpp -old_header=%t.dir/move-enum/enum.h %t.dir/move-enum/enum.cpp -- -std=c++11
// RUN: FileCheck -input-file=%t.dir/move-enum/new_test.h -check-prefix=CHECK-NEW-TEST-H-CASE2 %s
// RUN: FileCheck -input-file=%t.dir/move-enum/enum.h -check-prefix=CHECK-OLD-TEST-H-CASE2 %s

// CHECK-NEW-TEST-H-CASE2: namespace a {
// CHECK-NEW-TEST-H-CASE2-NEXT: enum class E2 { Yellow };
// CHECK-NEW-TEST-H-CASE2-NEXT: }

// CHECK-OLD-TEST-H-CASE2-NOT: enum class E2 { Yellow };


// -----------------------------------------------------------------------------
// Test not moving class-insided enum declarations.
// -----------------------------------------------------------------------------
// RUN: cp %S/Inputs/enum.h  %t.dir/move-enum/enum.h
// RUN: echo '#include "enum.h"' > %t.dir/move-enum/enum.cpp
// RUN: clang-move -names="a::C::E3" -new_cc=%t.dir/move-enum/new_test.cpp -new_header=%t.dir/move-enum/new_test.h -old_cc=%t.dir/move-enum/enum.cpp -old_header=%t.dir/move-enum/enum.h %t.dir/move-enum/enum.cpp -- -std=c++11
// RUN: FileCheck -input-file=%t.dir/move-enum/new_test.h -allow-empty -check-prefix=CHECK-EMPTY %s

// CHECK-EMPTY: {{^}}{{$}}
