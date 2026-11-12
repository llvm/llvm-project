// RUN: mkdir -p %t.dir/clang-move/build
// RUN: mkdir -p %t.dir/clang-move/include
// RUN: mkdir -p %t.dir/clang-move/src
// RUN: sed 's|$test_dir|%/t.dir/clang-move|g' %S/Inputs/database_template.json > %t.dir/clang-move/compile_commands.json
// RUN: cp %S/Inputs/test.h  %t.dir/clang-move/include
// RUN: cp %S/Inputs/test.cpp %t.dir/clang-move/src
// RUN: touch %t.dir/clang-move/include/test2.h
// RUN: cd %t.dir/clang-move/build
// RUN: clang-move -names="a::Foo" -new_cc=%t.dir/clang-move/new_test.cpp -new_header=%t.dir/clang-move/new_test.h -old_cc=../src/test.cpp -old_header=../include/test.h %t.dir/clang-move/src/test.cpp
// RUN: FileCheck -input-file=%t.dir/clang-move/new_test.cpp -check-prefix=CHECK-NEW-TEST-CPP %s
// RUN: FileCheck -input-file=%t.dir/clang-move/new_test.h -check-prefix=CHECK-NEW-TEST-H %s
// RUN: FileCheck -input-file=%t.dir/clang-move/src/test.cpp -check-prefix=CHECK-OLD-TEST-EMPTY -allow-empty %s
// RUN: FileCheck -input-file=%t.dir/clang-move/include/test.h -check-prefix=CHECK-OLD-TEST-EMPTY -allow-empty %s
//
// RUN: cp %S/Inputs/test.h  %t.dir/clang-move/include
// RUN: cp %S/Inputs/test.cpp %t.dir/clang-move/src
// RUN: cd %t.dir/clang-move/build
// RUN: clang-move -names="a::Foo" -new_cc=%t.dir/clang-move/new_test.cpp -new_header=%t.dir/clang-move/new_test.h -old_cc=%t.dir/clang-move/src/test.cpp -old_header=%t.dir/clang-move/include/test.h %t.dir/clang-move/src/test.cpp
// RUN: FileCheck -input-file=%t.dir/clang-move/new_test.cpp -check-prefix=CHECK-NEW-TEST-CPP %s
// RUN: FileCheck -input-file=%t.dir/clang-move/new_test.h -check-prefix=CHECK-NEW-TEST-H %s
// RUN: FileCheck -input-file=%t.dir/clang-move/src/test.cpp -check-prefix=CHECK-OLD-TEST-EMPTY -allow-empty %s
// RUN: FileCheck -input-file=%t.dir/clang-move/include/test.h -check-prefix=CHECK-OLD-TEST-EMPTY -allow-empty %s
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
