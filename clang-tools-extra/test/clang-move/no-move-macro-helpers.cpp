// RUN: mkdir -p %T/no-move-macro-helper
// RUN: cp %S/Inputs/macro_helper_test.h  %T/no-move-macro-helper/macro_helper_test.h
// RUN: cp %S/Inputs/macro_helper_test.cpp %T/no-move-macro-helper/macro_helper_test.cpp
// RUN: cd %T/no-move-macro-helper
//
// -----------------------------------------------------------------------------
// Test no moving helpers in macro.
// -----------------------------------------------------------------------------
// RUN: clang-move -names="A" -new_cc=%T/no-move-macro-helper/new_test.cpp -new_header=%T/no-move-macro-helper/new_test.h -old_cc=%T/no-move-macro-helper/macro_helper_test.cpp -old_header=%T/no-move-macro-helper/macro_helper_test.h %T/no-move-macro-helper/macro_helper_test.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/no-move-macro-helper/new_test.h -check-prefix=CHECK-NEW-TEST-CASE1-H %s
// RUN: FileCheck -input-file=%T/no-move-macro-helper/new_test.cpp -check-prefix=CHECK-NEW-TEST-CASE1-CPP %s
// RUN: FileCheck -input-file=%T/no-move-macro-helper/macro_helper_test.h -check-prefix=CHECK-OLD-TEST-CASE1-H %s
// RUN: FileCheck -input-file=%T/no-move-macro-helper/macro_helper_test.cpp -check-prefix=CHECK-OLD-TEST-CASE1-CPP %s

// CHECK-NEW-TEST-CASE1-H: class A {};

// CHECK-OLD-TEST-CASE1-H-NOT: class A {};

// CHECK-OLD-TEST-CASE1-CPP: DEFINE(test)

// CHECK-NEW-TEST-CASE1-CPP-NOT: DEFINE(test)


// -----------------------------------------------------------------------------
// Test moving all.
// -----------------------------------------------------------------------------
// RUN: cp %S/Inputs/macro_helper_test.h  %T/no-move-macro-helper/macro_helper_test.h
// RUN: cp %S/Inputs/macro_helper_test.cpp %T/no-move-macro-helper/macro_helper_test.cpp
// RUN: clang-move -names="A, f1" -new_cc=%T/no-move-macro-helper/new_test.cpp -new_header=%T/no-move-macro-helper/new_test.h -old_cc=%T/no-move-macro-helper/macro_helper_test.cpp -old_header=%T/no-move-macro-helper/macro_helper_test.h %T/no-move-macro-helper/macro_helper_test.cpp -- -std=c++11
//
// RUN: FileCheck -input-file=%T/no-move-macro-helper/new_test.h -check-prefix=CHECK-NEW-TEST-CASE2-H %s
// RUN: FileCheck -input-file=%T/no-move-macro-helper/new_test.cpp -check-prefix=CHECK-NEW-TEST-CASE2-CPP %s
// RUN: FileCheck -input-file=%T/no-move-macro-helper/macro_helper_test.h -allow-empty -check-prefix=CHECK-EMPTY %s
// RUN: FileCheck -input-file=%T/no-move-macro-helper/macro_helper_test.cpp -allow-empty -check-prefix=CHECK-EMPTY %s

// CHECK-NEW-TEST-CASE2-H: class A {};
// CHECK-NEW-TEST-CASE2-H-NEXT:void f1();


// CHECK-NEW-TEST-CASE2-CPP: DEFINE(test)
// CHECK-NEW-TEST-CASE2-CPP: void f1() {}

// CHECK-EMPTY: {{^}}{{$}}
