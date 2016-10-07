// RUN: rm -rf %T/clang-move
// RUN: mkdir -p %T/clang-move/build
// RUN: sed 's|$test_dir|%/T/clang-move|g' %S/Inputs/database_template.json > %T/clang-move/compile_commands.json
// RUN: cp %S/Inputs/multiple_class_test*  %T/clang-move/
// RUN: cd %T/clang-move
// RUN: clang-move -names="a::Move1, b::Move2,c::Move3,c::Move4" -new_cc=%T/clang-move/new_multiple_class_test.cpp -new_header=%T/clang-move/new_multiple_class_test.h -old_cc=%T/clang-move/multiple_class_test.cpp -old_header=../clang-move/multiple_class_test.h %T/clang-move/multiple_class_test.cpp
// RUN: FileCheck -input-file=%T/clang-move/new_multiple_class_test.cpp -check-prefix=CHECK-NEW-TEST-CPP %s
// RUN: FileCheck -input-file=%T/clang-move/new_multiple_class_test.h -check-prefix=CHECK-NEW-TEST-H %s
// RUN: FileCheck -input-file=%T/clang-move/multiple_class_test.cpp -check-prefix=CHECK-OLD-TEST-CPP %s
// RUN: FileCheck -input-file=%T/clang-move/multiple_class_test.h -check-prefix=CHECK-OLD-TEST-H %s
//
// CHECK-OLD-TEST-H: namespace c {
// CHECK-OLD-TEST-H: class NoMove {
// CHECK-OLD-TEST-H: public:
// CHECK-OLD-TEST-H:   int f();
// CHECK-OLD-TEST-H: };
// CHECK-OLD-TEST-H: } // namespace c

// CHECK-OLD-TEST-CPP: #include "{{.*}}multiple_class_test.h"
// CHECK-OLD-TEST-CPP: namespace c {
// CHECK-OLD-TEST-CPP: int NoMove::f() {
// CHECK-OLD-TEST-CPP:   return 0;
// CHECK-OLD-TEST-CPP: }
// CHECK-OLD-TEST-CPP: } // namespace c

// CHECK-NEW-TEST-H: namespace a {
// CHECK-NEW-TEST-H: class Move1 {
// CHECK-NEW-TEST-H: public:
// CHECK-NEW-TEST-H:   int f();
// CHECK-NEW-TEST-H: };
// CHECK-NEW-TEST-H: } // namespace a
// CHECK-NEW-TEST-H: namespace b {
// CHECK-NEW-TEST-H: class Move2 {
// CHECK-NEW-TEST-H: public:
// CHECK-NEW-TEST-H:   int f();
// CHECK-NEW-TEST-H: };
// CHECK-NEW-TEST-H: } // namespace b
// CHECK-NEW-TEST-H: namespace c {
// CHECK-NEW-TEST-H: class Move3 {
// CHECK-NEW-TEST-H: public:
// CHECK-NEW-TEST-H:   int f();
// CHECK-NEW-TEST-H: };
// CHECK-NEW-TEST-H: class Move4 {
// CHECK-NEW-TEST-H: public:
// CHECK-NEW-TEST-H:   int f();
// CHECK-NEW-TEST-H: };
// CHECK-NEW-TEST-H: } // namespace c

// CHECK-NEW-TEST-CPP: #include "{{.*}}new_multiple_class_test.h"
// CHECK-NEW-TEST-CPP: namespace a {
// CHECK-NEW-TEST-CPP: int Move1::f() { return 0; }
// CHECK-NEW-TEST-CPP: } // namespace a
// CHECK-NEW-TEST-CPP: namespace b {
// CHECK-NEW-TEST-CPP: int Move2::f() { return 0; }
// CHECK-NEW-TEST-CPP: } // namespace b
// CHECK-NEW-TEST-CPP: namespace c {
// CHECK-NEW-TEST-CPP: int Move3::f() { return 0; }
// CHECK-NEW-TEST-CPP: int Move4::f() { return 0; }
// CHECK-NEW-TEST-CPP: } // namespace c
