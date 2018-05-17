// UNSUPPORTED: system-windows
// new_test.cpp contains #include of old test.h when running on windows. This is
// probably by a bug for path handling in clang-move.
//
// RUN: mkdir -p %T/clang-move/build
// RUN: mkdir -p %T/clang-move/include
// RUN: mkdir -p %T/clang-move/src
// RUN: sed 's|$test_dir|%/T/clang-move|g' %S/Inputs/database_template.json > %T/clang-move/compile_commands.json
// RUN: cp %S/Inputs/test.h  %T/clang-move/include
// RUN: cp %S/Inputs/test.cpp %T/clang-move/src
// RUN: touch %T/clang-move/include/test2.h
// RUN: cd %T/clang-move/build
// RUN: clang-move -names="a::Foo" -new_cc=%T/clang-move/new_test.cpp -new_header=%T/clang-move/new_test.h -old_cc=../src/test.cpp -old_header=../include/test.h %T/clang-move/src/test.cpp
// RUN: FileCheck -input-file=%T/clang-move/new_test.cpp -check-prefix=CHECK-NEW-TEST-CPP %s
// RUN: FileCheck -input-file=%T/clang-move/new_test.h -check-prefix=CHECK-NEW-TEST-H %s
// RUN: FileCheck -input-file=%T/clang-move/src/test.cpp -check-prefix=CHECK-OLD-TEST-EMPTY -allow-empty %s
// RUN: FileCheck -input-file=%T/clang-move/include/test.h -check-prefix=CHECK-OLD-TEST-EMPTY -allow-empty %s
//
// RUN: cp %S/Inputs/test.h  %T/clang-move/include
// RUN: cp %S/Inputs/test.cpp %T/clang-move/src
// RUN: cd %T/clang-move/build
// RUN: clang-move -names="a::Foo" -new_cc=%T/clang-move/new_test.cpp -new_header=%T/clang-move/new_test.h -old_cc=%T/clang-move/src/test.cpp -old_header=%T/clang-move/include/test.h %T/clang-move/src/test.cpp
// RUN: FileCheck -input-file=%T/clang-move/new_test.cpp -check-prefix=CHECK-NEW-TEST-CPP %s
// RUN: FileCheck -input-file=%T/clang-move/new_test.h -check-prefix=CHECK-NEW-TEST-H %s
// RUN: FileCheck -input-file=%T/clang-move/src/test.cpp -check-prefix=CHECK-OLD-TEST-EMPTY -allow-empty %s
// RUN: FileCheck -input-file=%T/clang-move/include/test.h -check-prefix=CHECK-OLD-TEST-EMPTY -allow-empty %s
//
//
// CHECK-NEW-TEST-H: #ifndef TEST_H // comment 1
// CHECK-NEW-TEST-H: #define TEST_H
// CHECK-NEW-TEST-H: namespace a {
// CHECK-NEW-TEST-H: class Foo {
// CHECK-NEW-TEST-H: public:
// CHECK-NEW-TEST-H:   int f();
// CHECK-NEW-TEST-H:   int f2(int a, int b);
// CHECK-NEW-TEST-H: };
// CHECK-NEW-TEST-H: } // namespace a
// CHECK-NEW-TEST-H: #endif // TEST_H
//
// CHECK-NEW-TEST-CPP: #include "{{.*}}new_test.h"
// CHECK-NEW-TEST-CPP: #include "test2.h"
// CHECK-NEW-TEST-CPP: namespace a {
// CHECK-NEW-TEST-CPP: int Foo::f() { return 0; }
// CHECK-NEW-TEST-CPP: int Foo::f2(int a, int b) { return a + b; }
// CHECK-NEW-TEST-CPP: } // namespace a
//
// CHECK-OLD-TEST-EMPTY: {{^}}{{$}}
