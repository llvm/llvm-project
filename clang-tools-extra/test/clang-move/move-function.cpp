// RUN: mkdir -p %T/move-function
// RUN: cat %S/Inputs/function_test.h > %T/move-function/function_test.h
// RUN: cat %S/Inputs/function_test.cpp > %T/move-function/function_test.cpp
// RUN: cd %T/move-function
// RUN: clang-move -names="g" -new_header=%T/move-function/new_function_test.h -old_header=../move-function/function_test.h %T/move-function/function_test.cpp --
// RUN: FileCheck -input-file=%T/move-function/new_function_test.h -check-prefix=CHECK-NEW-TEST-H-CASE1 %s
//
// CHECK-NEW-TEST-H-CASE1: #ifndef {{.*}}NEW_FUNCTION_TEST_H
// CHECK-NEW-TEST-H-CASE1: #define {{.*}}NEW_FUNCTION_TEST_H
// CHECK-NEW-TEST-H-CASE1: {{[[:space:]]+}}
// CHECK-NEW-TEST-H-CASE1: inline int g() { return 0; }
// CHECK-NEW-TEST-H-CASE1: {{[[:space:]]+}}
// CHECK-NEW-TEST-H-CASE1: #endif // {{.*}}NEW_FUNCTION_TEST_H
//
// RUN: cp %S/Inputs/function_test*  %T/move-function
// RUN: clang-move -names="h" -new_header=%T/move-function/new_function_test.h -old_header=../move-function/function_test.h %T/move-function/function_test.cpp --
// RUN: FileCheck -input-file=%T/move-function/new_function_test.h -check-prefix=CHECK-NEW-TEST-H-CASE2 %s
//
// CHECK-NEW-TEST-H-CASE2: #ifndef {{.*}}NEW_FUNCTION_TEST_H
// CHECK-NEW-TEST-H-CASE2: #define {{.*}}NEW_FUNCTION_TEST_H
// CHECK-NEW-TEST-H-CASE2: {{[[:space:]]+}}
// CHECK-NEW-TEST-H-CASE2: template <typename T> void h(T t) {}
// CHECK-NEW-TEST-H-CASE2: {{[[:space:]]+}}
// CHECK-NEW-TEST-H-CASE2: template <> void h(int t) {}
// CHECK-NEW-TEST-H-CASE2: {{[[:space:]]+}}
// CHECK-NEW-TEST-H-CASE2: #endif // {{.*}}NEW_FUNCTION_TEST_H
//
// RUN: cp %S/Inputs/function_test*  %T/move-function
// RUN: clang-move -names="f" -new_header=%T/move-function/new_function_test.h -new_cc=%T/move-function/new_function_test.cpp -old_header=../move-function/function_test.h -old_cc=../move-function/function_test.cpp %T/move-function/function_test.cpp --
// RUN: FileCheck -input-file=%T/move-function/new_function_test.h -check-prefix=CHECK-NEW-TEST-H-CASE3 %s
// RUN: FileCheck -input-file=%T/move-function/new_function_test.cpp -check-prefix=CHECK-NEW-TEST-CPP-CASE3 %s
//
// CHECK-NEW-TEST-H-CASE3: #ifndef {{.*}}NEW_FUNCTION_TEST_H
// CHECK-NEW-TEST-H-CASE3: #define {{.*}}NEW_FUNCTION_TEST_H
// CHECK-NEW-TEST-H-CASE3: {{[[:space:]]+}}
// CHECK-NEW-TEST-H-CASE3: void f();
// CHECK-NEW-TEST-H-CASE3: {{[[:space:]]+}}
// CHECK-NEW-TEST-H-CASE3: #endif // {{.*}}NEW_FUNCTION_TEST_H
// CHECK-NEW-TEST-CPP-CASE3: #include "{{.*}}new_function_test.h"
// CHECK-NEW-TEST-CPP-CASE3: {{[[:space:]]+}}
// CHECK-NEW-TEST-CPP-CASE3: void f() {}
//
// RUN: cat %S/Inputs/function_test.h > %T/move-function/function_test.h
// RUN: cat %S/Inputs/function_test.cpp > %T/move-function/function_test.cpp
// RUN: clang-move -names="A::f" -new_header=%T/move-function/new_function_test.h -new_cc=%T/move-function/new_function_test.cpp -old_header=../move-function/function_test.h -old_cc=../move-function/function_test.cpp %T/move-function/function_test.cpp -dump_result -- | FileCheck %s -check-prefix=CHECK-EMPTY
//
// CHECK-EMPTY: [{{[[:space:]]*}}]
//
// RUN: cat %S/Inputs/function_test.h > %T/move-function/function_test.h
// RUN: cat %S/Inputs/function_test.cpp > %T/move-function/function_test.cpp
// RUN: clang-move -names="f,A" -new_header=%T/move-function/new_function_test.h -new_cc=%T/move-function/new_function_test.cpp -old_header=../move-function/function_test.h -old_cc=../move-function/function_test.cpp %T/move-function/function_test.cpp --
// RUN: FileCheck -input-file=%T/move-function/new_function_test.h -check-prefix=CHECK-NEW-TEST-H-CASE4 %s
// RUN: FileCheck -input-file=%T/move-function/new_function_test.cpp -check-prefix=CHECK-NEW-TEST-CPP-CASE4 %s

// CHECK-NEW-TEST-H-CASE4: #ifndef {{.*}}NEW_FUNCTION_TEST_H
// CHECK-NEW-TEST-H-CASE4: #define {{.*}}NEW_FUNCTION_TEST_H
// CHECK-NEW-TEST-H-CASE4: {{[[:space:]]+}}
// CHECK-NEW-TEST-H-CASE4: void f();
// CHECK-NEW-TEST-H-CASE4: {{[[:space:]]+}}
// CHECK-NEW-TEST-H-CASE4: class A {
// CHECK-NEW-TEST-H-CASE4: public:
// CHECK-NEW-TEST-H-CASE4:   void f();
// CHECK-NEW-TEST-H-CASE4: };
// CHECK-NEW-TEST-H-CASE4: {{[[:space:]]+}}
// CHECK-NEW-TEST-H-CASE4: #endif // {{.*}}NEW_FUNCTION_TEST_H
// CHECK-NEW-TEST-CPP-CASE4: #include "{{.*}}new_function_test.h"
// CHECK-NEW-TEST-CPP-CASE4: {{[[:space:]]+}}
// CHECK-NEW-TEST-CPP-CASE4: void f() {}
// CHECK-NEW-TEST-CPP-CASE4: {{[[:space:]]+}}
// CHECK-NEW-TEST-CPP-CASE4: void A::f() {}
