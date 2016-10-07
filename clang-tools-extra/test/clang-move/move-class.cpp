// RUN: rm -rf %T/clang-move
// RUN: mkdir -p %T/clang-move/build
// RUN: sed 's|$test_dir|%/T/clang-move|g' %S/Inputs/database_template.json > %T/clang-move/compile_commands.json
// RUN: cp %S/Inputs/test*  %T/clang-move/
// RUN: touch %T/clang-move/test2.h
// RUN: cd %T/clang-move
// RUN: clang-move -names="a::Foo" -new_cc=%T/clang-move/new_test.cpp -new_header=%T/clang-move/new_test.h -old_cc=../clang-move/test.cpp -old_header=../clang-move/test.h %T/clang-move/test.cpp
// RUN: FileCheck -input-file=%T/clang-move/new_test.cpp -check-prefix=CHECK-NEW-TEST-CPP %s
// RUN: FileCheck -input-file=%T/clang-move/new_test.h -check-prefix=CHECK-NEW-TEST-H %s
// RUN: FileCheck -input-file=%T/clang-move/test.cpp -check-prefix=CHECK-OLD-TEST-CPP %s
// RUN: FileCheck -input-file=%T/clang-move/test.h %s -implicit-check-not='{{namespace.*}}'
//
// RUN: cp %S/Inputs/test*  %T/clang-move/
// RUN: cd %T/clang-move
// RUN: clang-move -names="a::Foo" -new_cc=%T/clang-move/new_test.cpp -new_header=%T/clang-move/new_test.h -old_cc=%T/clang-move/test.cpp -old_header=%T/clang-move/test.h %T/clang-move/test.cpp
// RUN: FileCheck -input-file=%T/clang-move/new_test.cpp -check-prefix=CHECK-NEW-TEST-CPP %s
// RUN: FileCheck -input-file=%T/clang-move/new_test.h -check-prefix=CHECK-NEW-TEST-H %s
// RUN: FileCheck -input-file=%T/clang-move/test.cpp -check-prefix=CHECK-OLD-TEST-CPP %s
// RUN: FileCheck -input-file=%T/clang-move/test.h %s -implicit-check-not='{{namespace.*}}'
//
// CHECK-NEW-TEST-H: namespace a {
// CHECK-NEW-TEST-H: class Foo {
// CHECK-NEW-TEST-H: public:
// CHECK-NEW-TEST-H:   int f();
// CHECK-NEW-TEST-H: };
// CHECK-NEW-TEST-H: } // namespace a
//
// CHECK-NEW-TEST-CPP: #include "{{.*}}new_test.h"
// CHECK-NEW-TEST-CPP: #include "test2.h"
// CHECK-NEW-TEST-CPP: namespace a {
// CHECK-NEW-TEST-CPP: int Foo::f() { return 0; }
// CHECK-NEW-TEST-CPP: } // namespace a
//
// CHECK-OLD-TEST-CPP: #include "test.h"
// CHECK-OLD-TEST-CPP: #include "test2.h"
