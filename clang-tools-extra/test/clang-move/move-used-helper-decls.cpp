// RUN: mkdir -p %T/used-helper-decls
// RUN: cp %S/Inputs/helper_decls_test*  %T/used-helper-decls/
// RUN: cd %T/used-helper-decls

// ----------------------------------------------------------------------------
// Test moving used helper function and its transitively used functions.
// ----------------------------------------------------------------------------
// RUN: clang-move -names="a::Class1" -new_cc=%T/used-helper-decls/new_helper_decls_test.cpp -new_header=%T/used-helper-decls/new_helper_decls_test.h -old_cc=%T/used-helper-decls/helper_decls_test.cpp -old_header=../used-helper-decls/helper_decls_test.h %T/used-helper-decls/helper_decls_test.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/used-helper-decls/new_helper_decls_test.cpp -check-prefix=CHECK-NEW-CLASS1-CPP %s
// RUN: FileCheck -input-file=%T/used-helper-decls/helper_decls_test.cpp -check-prefix=CHECK-OLD-CLASS1-CPP %s

// CHECK-NEW-CLASS1-CPP: #include "{{.*}}new_helper_decls_test.h"
// CHECK-NEW-CLASS1-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS1-CPP-NEXT: namespace {
// CHECK-NEW-CLASS1-CPP-NEXT: void HelperFun1() {}
// CHECK-NEW-CLASS1-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS1-CPP-NEXT: void HelperFun2() { HelperFun1(); }
// CHECK-NEW-CLASS1-CPP-NEXT: } // namespace
// CHECK-NEW-CLASS1-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS1-CPP-NEXT: namespace a {
// CHECK-NEW-CLASS1-CPP-NEXT: void Class1::f() { HelperFun2(); }
// CHECK-NEW-CLASS1-CPP-NEXT: } // namespace a
//
// CHECK-OLD-CLASS1-CPP: void HelperFun1() {}
// CHECK-OLD-CLASS1-CPP-NOT: void HelperFun2() { HelperFun1(); }
// CHECK-OLD-CLASS1-CPP-NOT: void Class1::f() { HelperFun2(); }
// CHECK-OLD-CLASS1-CPP: void Class2::f() {
// CHECK-OLD-CLASS1-CPP:   HelperFun1();


// ----------------------------------------------------------------------------
// Test moving used helper function and its transitively used static variables.
// ----------------------------------------------------------------------------
// RUN: cp %S/Inputs/helper_decls_test*  %T/used-helper-decls/
// RUN: clang-move -names="a::Class2" -new_cc=%T/used-helper-decls/new_helper_decls_test.cpp -new_header=%T/used-helper-decls/new_helper_decls_test.h -old_cc=%T/used-helper-decls/helper_decls_test.cpp -old_header=../used-helper-decls/helper_decls_test.h %T/used-helper-decls/helper_decls_test.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/used-helper-decls/new_helper_decls_test.cpp -check-prefix=CHECK-NEW-CLASS2-CPP %s
// RUN: FileCheck -input-file=%T/used-helper-decls/helper_decls_test.cpp -check-prefix=CHECK-OLD-CLASS2-CPP %s

// CHECK-NEW-CLASS2-CPP: #include "{{.*}}new_helper_decls_test.h"
// CHECK-NEW-CLASS2-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS2-CPP-NEXT: namespace {
// CHECK-NEW-CLASS2-CPP-NEXT: void HelperFun1() {}
// CHECK-NEW-CLASS2-CPP-NEXT: } // namespace
// CHECK-NEW-CLASS2-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS2-CPP-NEXT: static const int K2 = 2;
// CHECK-NEW-CLASS2-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS2-CPP-NEXT: static void HelperFun3() { K2; }
// CHECK-NEW-CLASS2-CPP-NEXT: namespace a {
// CHECK-NEW-CLASS2-CPP-NEXT: void Class2::f() {
// CHECK-NEW-CLASS2-CPP-NEXT:   HelperFun1();
// CHECK-NEW-CLASS2-CPP-NEXT:   HelperFun3();
// CHECK-NEW-CLASS2-CPP-NEXT: }
// CHECK-NEW-CLASS2-CPP-NEXT: } // namespace a

// CHECK-OLD-CLASS2-CPP: void HelperFun1() {}
// CHECK-OLD-CLASS2-CPP: void HelperFun2() { HelperFun1(); }
// CHECK-OLD-CLASS2-CPP: const int K1 = 1;
// CHECK-OLD-CLASS2-CPP: static const int K2 = 2;
// CHECK-OLD-CLASS2-CPP-NOT: static void HelperFun3() { K2; }
// CHECK-OLD-CLASS2-CPP-NOT: void Class2::f() {
// CHECK-OLD-CLASS2-CPP-NOT:   HelperFun1();
// CHECK-OLD-CLASS2-CPP-NOT:   HelperFun3();
// CHECK-OLD-CLASS2-CPP: void Class5::f() {
// CHECK-OLD-CLASS2-CPP-NEXT: int Result = K1 + K2 + K3;


// ----------------------------------------------------------------------------
// Test using a static member variable of a helper class.
// ----------------------------------------------------------------------------
// RUN: cp %S/Inputs/helper_decls_test*  %T/used-helper-decls/
// RUN: clang-move -names="a::Class3" -new_cc=%T/used-helper-decls/new_helper_decls_test.cpp -new_header=%T/used-helper-decls/new_helper_decls_test.h -old_cc=%T/used-helper-decls/helper_decls_test.cpp -old_header=../used-helper-decls/helper_decls_test.h %T/used-helper-decls/helper_decls_test.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/used-helper-decls/new_helper_decls_test.cpp -check-prefix=CHECK-NEW-CLASS3-CPP %s
// RUN: FileCheck -input-file=%T/used-helper-decls/helper_decls_test.cpp -check-prefix=CHECK-OLD-CLASS3-CPP %s

// CHECK-NEW-CLASS3-CPP: #include "{{.*}}new_helper_decls_test.h"
// CHECK-NEW-CLASS3-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS3-CPP-NEXT: namespace {
// CHECK-NEW-CLASS3-CPP-NEXT: class HelperC1 {
// CHECK-NEW-CLASS3-CPP-NEXT: public:
// CHECK-NEW-CLASS3-CPP-NEXT:   static int I;
// CHECK-NEW-CLASS3-CPP-NEXT: };
// CHECK-NEW-CLASS3-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS3-CPP-NEXT: int HelperC1::I = 0;
// CHECK-NEW-CLASS3-CPP-NEXT: } // namespace
// CHECK-NEW-CLASS3-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS3-CPP-NEXT: namespace a {
// CHECK-NEW-CLASS3-CPP-NEXT: void Class3::f() { HelperC1::I; }
// CHECK-NEW-CLASS3-CPP-NEXT: } // namespace a

// CHECK-OLD-CLASS3-CPP: namespace {
// CHECK-OLD-CLASS3-CPP-NOT: class HelperC1 {
// CHECK-OLD-CLASS3-CPP-NOT: public:
// CHECK-OLD-CLASS3-CPP-NOT:   static int I;
// CHECK-OLD-CLASS3-CPP-NOT: };
// CHECK-OLD-CLASS3-CPP-NOT: int HelperC1::I = 0;
// CHECK-OLD-CLASS3-CPP: class HelperC2 {};


// ----------------------------------------------------------------------------
// Test moving helper classes.
// ----------------------------------------------------------------------------
// RUN: cp %S/Inputs/helper_decls_test*  %T/used-helper-decls/
// RUN: clang-move -names="a::Class4" -new_cc=%T/used-helper-decls/new_helper_decls_test.cpp -new_header=%T/used-helper-decls/new_helper_decls_test.h -old_cc=%T/used-helper-decls/helper_decls_test.cpp -old_header=../used-helper-decls/helper_decls_test.h %T/used-helper-decls/helper_decls_test.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/used-helper-decls/new_helper_decls_test.cpp -check-prefix=CHECK-NEW-CLASS4-CPP %s
// RUN: FileCheck -input-file=%T/used-helper-decls/helper_decls_test.cpp -check-prefix=CHECK-OLD-CLASS4-CPP %s

// CHECK-NEW-CLASS4-CPP: #include "{{.*}}new_helper_decls_test.h"
// CHECK-NEW-CLASS4-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS4-CPP-NEXT: namespace {
// CHECK-NEW-CLASS4-CPP-NEXT: class HelperC2 {};
// CHECK-NEW-CLASS4-CPP-NEXT: } // namespace
// CHECK-NEW-CLASS4-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS4-CPP-NEXT: namespace a {
// CHECK-NEW-CLASS4-CPP-NEXT: void Class4::f() { HelperC2 c2; }
// CHECK-NEW-CLASS4-CPP-NEXT: } // namespace a

// CHECK-OLD-CLASS4-CPP-NOT: class HelperC2 {};


// ----------------------------------------------------------------------------
// Test moving helper variables and helper functions together.
// ----------------------------------------------------------------------------
// RUN: cp %S/Inputs/helper_decls_test*  %T/used-helper-decls/
// RUN: clang-move -names="a::Class5" -new_cc=%T/used-helper-decls/new_helper_decls_test.cpp -new_header=%T/used-helper-decls/new_helper_decls_test.h -old_cc=%T/used-helper-decls/helper_decls_test.cpp -old_header=../used-helper-decls/helper_decls_test.h %T/used-helper-decls/helper_decls_test.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/used-helper-decls/new_helper_decls_test.cpp -check-prefix=CHECK-NEW-CLASS5-CPP %s
// RUN: FileCheck -input-file=%T/used-helper-decls/helper_decls_test.cpp -check-prefix=CHECK-OLD-CLASS5-CPP %s

// CHECK-NEW-CLASS5-CPP: #include "{{.*}}new_helper_decls_test.h"
// CHECK-NEW-CLASS5-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS5-CPP-NEXT: namespace {
// CHECK-NEW-CLASS5-CPP-NEXT: const int K1 = 1;
// CHECK-NEW-CLASS5-CPP-NEXT: } // namespace
// CHECK-NEW-CLASS5-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS5-CPP-NEXT: static const int K2 = 2;
// CHECK-NEW-CLASS5-CPP-NEXT: namespace a {
// CHECK-NEW-CLASS5-CPP-NEXT: static const int K3 = 3;
// CHECK-NEW-CLASS5-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS5-CPP-NEXT: static void HelperFun4() {}
// CHECK-NEW-CLASS5-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS5-CPP-NEXT: void Class5::f() {
// CHECK-NEW-CLASS5-CPP-NEXT:   int Result = K1 + K2 + K3;
// CHECK-NEW-CLASS5-CPP-NEXT:   HelperFun4();
// CHECK-NEW-CLASS5-CPP-NEXT: }
// CHECK-NEW-CLASS5-CPP-NEXT: } // namespace a

// CHECK-OLD-CLASS5-CPP-NOT: const int K1 = 1;
// CHECK-OLD-CLASS5-CPP: static const int K2 = 2;
// CHECK-OLD-CLASS5-CPP: static void HelperFun3() { K2; }
// CHECK-OLD-CLASS5-CPP: static const int K4 = HelperC3::I;
// CHECK-OLD-CLASS5-CPP-NOT: void Class5::f() {


// ----------------------------------------------------------------------------
// Test moving helper variables and their transitively used helper classes.
// ----------------------------------------------------------------------------
// RUN: cp %S/Inputs/helper_decls_test*  %T/used-helper-decls/
// RUN: clang-move -names="a::Class6" -new_cc=%T/used-helper-decls/new_helper_decls_test.cpp -new_header=%T/used-helper-decls/new_helper_decls_test.h -old_cc=%T/used-helper-decls/helper_decls_test.cpp -old_header=../used-helper-decls/helper_decls_test.h %T/used-helper-decls/helper_decls_test.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/used-helper-decls/new_helper_decls_test.cpp -check-prefix=CHECK-NEW-CLASS6-CPP %s
// RUN: FileCheck -input-file=%T/used-helper-decls/helper_decls_test.cpp -check-prefix=CHECK-OLD-CLASS6-CPP %s

// CHECK-NEW-CLASS6-CPP: #include "{{.*}}new_helper_decls_test.h"
// CHECK-NEW-CLASS6-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS6-CPP-NEXT: namespace {
// CHECK-NEW-CLASS6-CPP-NEXT: class HelperC3 {
// CHECK-NEW-CLASS6-CPP-NEXT: public:
// CHECK-NEW-CLASS6-CPP-NEXT:   static int I;
// CHECK-NEW-CLASS6-CPP-NEXT: };
// CHECK-NEW-CLASS6-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS6-CPP-NEXT: int HelperC3::I = 0;
// CHECK-NEW-CLASS6-CPP-NEXT: } // namespace
// CHECK-NEW-CLASS6-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS6-CPP-NEXT: namespace a {
// CHECK-NEW-CLASS6-CPP-NEXT: static const int K4 = HelperC3::I;
// CHECK-NEW-CLASS6-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS6-CPP-NEXT: int Class6::f() {
// CHECK-NEW-CLASS6-CPP-NEXT:   int R = K4;
// CHECK-NEW-CLASS6-CPP-NEXT:   return R;
// CHECK-NEW-CLASS6-CPP-NEXT: }
// CHECK-NEW-CLASS6-CPP-NEXT: } // namespace a

// CHECK-OLD-CLASS6-CPP-NOT: class HelperC3 {
// CHECK-OLD-CLASS6-CPP-NOT: int HelperC3::I = 0;
// CHECK-OLD-CLASS6-CPP-NOT: static const int K4 = HelperC3::I;


// ----------------------------------------------------------------------------
// Test moving classes where its methods use helpers.
// ----------------------------------------------------------------------------
// RUN: cp %S/Inputs/helper_decls_test*  %T/used-helper-decls/
// RUN: clang-move -names="a::Class7" -new_cc=%T/used-helper-decls/new_helper_decls_test.cpp -new_header=%T/used-helper-decls/new_helper_decls_test.h -old_cc=%T/used-helper-decls/helper_decls_test.cpp -old_header=../used-helper-decls/helper_decls_test.h %T/used-helper-decls/helper_decls_test.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/used-helper-decls/new_helper_decls_test.cpp -check-prefix=CHECK-NEW-CLASS7-CPP %s
// RUN: FileCheck -input-file=%T/used-helper-decls/helper_decls_test.cpp -check-prefix=CHECK-OLD-CLASS7-CPP %s

// CHECK-NEW-CLASS7-CPP: #include "{{.*}}new_helper_decls_test.h"
// CHECK-NEW-CLASS7-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS7-CPP-NEXT: namespace a {
// CHECK-NEW-CLASS7-CPP-NEXT: static const int K6 = 6;
// CHECK-NEW-CLASS7-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS7-CPP-NEXT: static void HelperFun6() {}
// CHECK-NEW-CLASS7-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS7-CPP-NEXT: int Class7::f() {
// CHECK-NEW-CLASS7-CPP-NEXT:   int R = K6;
// CHECK-NEW-CLASS7-CPP-NEXT:   return R;
// CHECK-NEW-CLASS7-CPP-NEXT: }
// CHECK-NEW-CLASS7-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CLASS7-CPP-NEXT: int Class7::g() {
// CHECK-NEW-CLASS7-CPP-NEXT:   HelperFun6();
// CHECK-NEW-CLASS7-CPP-NEXT:   return 1;
// CHECK-NEW-CLASS7-CPP-NEXT: }
// CHECK-NEW-CLASS7-CPP-NEXT: } // namespace a
//
// CHECK-OLD-CLASS7-CPP-NOT: static const int K6 = 6;
// CHECK-OLD-CLASS7-CPP-NOT: static void HelperFun6() {}
// CHECK-OLD-CLASS7-CPP-NOT: int Class7::f() {
// CHECK-OLD-CLASS7-CPP-NOT: int Class7::g() {


// ----------------------------------------------------------------------------
// Test moving helper function and its transitively used helper variables.
// ----------------------------------------------------------------------------
// RUN: cp %S/Inputs/helper_decls_test*  %T/used-helper-decls/
// RUN: clang-move -names="a::Fun1" -new_cc=%T/used-helper-decls/new_helper_decls_test.cpp -new_header=%T/used-helper-decls/new_helper_decls_test.h -old_cc=%T/used-helper-decls/helper_decls_test.cpp -old_header=../used-helper-decls/helper_decls_test.h %T/used-helper-decls/helper_decls_test.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/used-helper-decls/new_helper_decls_test.cpp -check-prefix=CHECK-NEW-FUN1-CPP %s
// RUN: FileCheck -input-file=%T/used-helper-decls/helper_decls_test.cpp -check-prefix=CHECK-OLD-FUN1-CPP %s

// CHECK-NEW-FUN1-CPP: #include "{{.*}}new_helper_decls_test.h"
// CHECK-NEW-FUN1-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-FUN1-CPP-NEXT: namespace a {
// CHECK-NEW-FUN1-CPP-NEXT: static const int K5 = 5;
// CHECK-NEW-FUN1-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-FUN1-CPP-NEXT: static int HelperFun5() {
// CHECK-NEW-FUN1-CPP-NEXT:   int R = K5;
// CHECK-NEW-FUN1-CPP-NEXT:   return R;
// CHECK-NEW-FUN1-CPP-NEXT: }
// CHECK-NEW-FUN1-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-FUN1-CPP-NEXT: void Fun1() { HelperFun5(); }
// CHECK-NEW-FUN1-CPP-NEXT: } // namespace a

// CHECK-OLD-FUN1-CPP-NOT: static const int K5 = 5;
// CHECK-OLD-FUN1-CPP-NOT: static int HelperFun5() {
// CHECK-OLD-FUN1-CPP-NOT: void Fun1() { HelperFun5(); }


// ----------------------------------------------------------------------------
// Test no moving helpers when moving inline functions in header.
// ----------------------------------------------------------------------------
// RUN: cp %S/Inputs/helper_decls_test*  %T/used-helper-decls/
// RUN: clang-move -names="a::Fun2" -new_cc=%T/used-helper-decls/new_helper_decls_test.cpp -new_header=%T/used-helper-decls/new_helper_decls_test.h -old_cc=%T/used-helper-decls/helper_decls_test.cpp -old_header=../used-helper-decls/helper_decls_test.h %T/used-helper-decls/helper_decls_test.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/used-helper-decls/new_helper_decls_test.cpp -check-prefix=CHECK-NEW-FUN2-CPP %s
// RUN: FileCheck -input-file=%T/used-helper-decls/new_helper_decls_test.h -check-prefix=CHECK-NEW-FUN2-H %s
// RUN: FileCheck -input-file=%T/used-helper-decls/helper_decls_test.h -check-prefix=CHECK-OLD-FUN2-H %s

// CHECK-NEW-FUN2-H: namespace a {
// CHECK-NEW-FUN2-H-NEXT: inline void Fun2() {}
// CHECK-NEW-FUN2-H-NEXT: } // namespace a

// CHECK-NEW-FUN2-CPP: #include "{{.*}}new_helper_decls_test.h"
// CHECK-NEW-FUN2-CPP-SAME: {{[[:space:]]}}

// CHECK-OLD-FUN2-H-NOT: inline void Fun2() {}

// ----------------------------------------------------------------------------
// Test moving used helper function and its transitively used functions.
// ----------------------------------------------------------------------------
// RUN: cp %S/Inputs/helper_decls_test*  %T/used-helper-decls/
// RUN: clang-move -names="b::Fun3" -new_cc=%T/used-helper-decls/new_helper_decls_test.cpp -new_header=%T/used-helper-decls/new_helper_decls_test.h -old_cc=%T/used-helper-decls/helper_decls_test.cpp -old_header=../used-helper-decls/helper_decls_test.h %T/used-helper-decls/helper_decls_test.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/used-helper-decls/new_helper_decls_test.cpp -check-prefix=CHECK-NEW-FUN3-CPP %s
// RUN: FileCheck -input-file=%T/used-helper-decls/helper_decls_test.cpp -check-prefix=CHECK-OLD-FUN3-CPP %s

// CHECK-NEW-FUN3-CPP: #include "{{.*}}new_helper_decls_test.h"
// CHECK-NEW-FUN3-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-FUN3-CPP-NEXT: namespace b {
// CHECK-NEW-FUN3-CPP-NEXT: namespace {
// CHECK-NEW-FUN3-CPP-NEXT: void HelperFun7();
// CHECK-NEW-FUN3-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-FUN3-CPP-NEXT: class HelperC4;
// CHECK-NEW-FUN3-CPP-NEXT: } // namespace
// CHECK-NEW-FUN3-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-FUN3-CPP-NEXT: void Fun3() {
// CHECK-NEW-FUN3-CPP-NEXT:   HelperFun7();
// CHECK-NEW-FUN3-CPP-NEXT:   HelperC4 *t;
// CHECK-NEW-FUN3-CPP-NEXT: }
// CHECK-NEW-FUN3-CPP-NEXT: namespace {
// CHECK-NEW-FUN3-CPP-NEXT: void HelperFun7() {}
// CHECK-NEW-FUN3-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-FUN3-CPP-NEXT: class HelperC4 {};
// CHECK-NEW-FUN3-CPP-NEXT: } // namespace
// CHECK-NEW-FUN3-CPP-NEXT: } // namespace b
//
// CHECK-OLD-FUN3-CPP-NOT: void HelperFun7();
// CHECK-OLD-FUN3-CPP-NOT: void HelperFun7() {}
// CHECK-OLD-FUN3-CPP-NOT: void Fun3() { HelperFun7(); }

// ----------------------------------------------------------------------------
// Test moving all symbols in headers.
// ----------------------------------------------------------------------------
// RUN: cp %S/Inputs/helper_decls_test*  %T/used-helper-decls/
// RUN: clang-move -names="a::Class1, a::Class2, a::Class3, a::Class4, a::Class5, a::Class5, a::Class6, a::Class7, a::Fun1, a::Fun2, b::Fun3" -new_cc=%T/used-helper-decls/new_helper_decls_test.cpp -new_header=%T/used-helper-decls/new_helper_decls_test.h -old_cc=%T/used-helper-decls/helper_decls_test.cpp -old_header=../used-helper-decls/helper_decls_test.h %T/used-helper-decls/helper_decls_test.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/used-helper-decls/new_helper_decls_test.h -check-prefix=CHECK-NEW-H %s
// RUN: FileCheck -input-file=%T/used-helper-decls/new_helper_decls_test.cpp -check-prefix=CHECK-NEW-CPP %s
// RUN: FileCheck -input-file=%T/used-helper-decls/helper_decls_test.h -allow-empty -check-prefix=CHECK-EMPTY %s
// RUN: FileCheck -input-file=%T/used-helper-decls/helper_decls_test.cpp -allow-empty -check-prefix=CHECK-EMPTY %s


// CHECK-NEW-H: namespace a {
// CHECK-NEW-H-NEXT: class Class1 {
// CHECK-NEW-H-NEXT:   void f();
// CHECK-NEW-H-NEXT: };
// CHECK-NEW-H-SAME: {{[[:space:]]}}
// CHECK-NEW-H-NEXT: class Class2 {
// CHECK-NEW-H-NEXT:   void f();
// CHECK-NEW-H-NEXT: };
// CHECK-NEW-H-SAME: {{[[:space:]]}}
// CHECK-NEW-H-NEXT: class Class3 {
// CHECK-NEW-H-NEXT:   void f();
// CHECK-NEW-H-NEXT: };
// CHECK-NEW-H-SAME: {{[[:space:]]}}
// CHECK-NEW-H-NEXT: class Class4 {
// CHECK-NEW-H-NEXT:   void f();
// CHECK-NEW-H-NEXT: };
// CHECK-NEW-H-SAME: {{[[:space:]]}}
// CHECK-NEW-H-NEXT: class Class5 {
// CHECK-NEW-H-NEXT:   void f();
// CHECK-NEW-H-NEXT: };
// CHECK-NEW-H-SAME: {{[[:space:]]}}
// CHECK-NEW-H-NEXT: class Class6 {
// CHECK-NEW-H-NEXT:   int f();
// CHECK-NEW-H-NEXT: };
// CHECK-NEW-H-SAME: {{[[:space:]]}}
// CHECK-NEW-H-NEXT: class Class7 {
// CHECK-NEW-H-NEXT:   int f();
// CHECK-NEW-H-NEXT:   int g();
// CHECK-NEW-H-NEXT: };
// CHECK-NEW-H-SAME: {{[[:space:]]}}
// CHECK-NEW-H-NEXT: void Fun1();
// CHECK-NEW-H-SAME: {{[[:space:]]}}
// CHECK-NEW-H-NEXT: inline void Fun2() {}
// CHECK-NEW-H-SAME: {{[[:space:]]}}
// CHECK-NEW-H-NEXT: } // namespace a


// CHECK-NEW-CPP: namespace {
// CHECK-NEW-CPP-NEXT: class HelperC1 {
// CHECK-NEW-CPP-NEXT: public:
// CHECK-NEW-CPP-NEXT:   static int I;
// CHECK-NEW-CPP-NEXT: };
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: int HelperC1::I = 0;
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: class HelperC2 {};
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: class HelperC3 {
// CHECK-NEW-CPP-NEXT:  public:
// CHECK-NEW-CPP-NEXT:   static int I;
// CHECK-NEW-CPP-NEXT: };
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: int HelperC3::I = 0;
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: void HelperFun1() {}
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: void HelperFun2() { HelperFun1(); }
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: const int K1 = 1;
// CHECK-NEW-CPP-NEXT: } // namespace
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: static const int K2 = 2;
// CHECK-NEW-CPP-NEXT: static void HelperFun3() { K2; }
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: namespace a {
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: static const int K3 = 3;
// CHECK-NEW-CPP-NEXT: static const int K4 = HelperC3::I;
// CHECK-NEW-CPP-NEXT: static const int K5 = 5;
// CHECK-NEW-CPP-NEXT: static const int K6 = 6;
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: static void HelperFun4() {}
// CHECK-NEW-CPP-NEXT: static void HelperFun6() {}
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: void Class1::f() { HelperFun2(); }
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: void Class2::f() {
// CHECK-NEW-CPP-NEXT:   HelperFun1();
// CHECK-NEW-CPP-NEXT:   HelperFun3();
// CHECK-NEW-CPP-NEXT: }
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: void Class3::f() { HelperC1::I; }
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: void Class4::f() { HelperC2 c2; }
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: void Class5::f() {
// CHECK-NEW-CPP-NEXT:   int Result = K1 + K2 + K3;
// CHECK-NEW-CPP-NEXT:   HelperFun4();
// CHECK-NEW-CPP-NEXT: }
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: int Class6::f() {
// CHECK-NEW-CPP-NEXT:   int R = K4;
// CHECK-NEW-CPP-NEXT:   return R;
// CHECK-NEW-CPP-NEXT: }
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: int Class7::f() {
// CHECK-NEW-CPP-NEXT:   int R = K6;
// CHECK-NEW-CPP-NEXT:   return R;
// CHECK-NEW-CPP-NEXT: }
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: int Class7::g() {
// CHECK-NEW-CPP-NEXT:   HelperFun6();
// CHECK-NEW-CPP-NEXT:   return 1;
// CHECK-NEW-CPP-NEXT: }
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: static int HelperFun5() {
// CHECK-NEW-CPP-NEXT:   int R = K5;
// CHECK-NEW-CPP-NEXT:   return R;
// CHECK-NEW-CPP-NEXT: }
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: void Fun1() { HelperFun5(); }
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: } // namespace a
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: namespace b {
// CHECK-NEW-CPP-NEXT: namespace {
// CHECK-NEW-CPP-NEXT: void HelperFun7();
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: class HelperC4;
// CHECK-NEW-CPP-NEXT: } // namespace
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: void Fun3() {
// CHECK-NEW-CPP-NEXT:   HelperFun7();
// CHECK-NEW-CPP-NEXT:   HelperC4 *t;
// CHECK-NEW-CPP-NEXT: }
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: namespace {
// CHECK-NEW-CPP-NEXT: void HelperFun7() {}
// CHECK-NEW-CPP-SAME: {{[[:space:]]}}
// CHECK-NEW-CPP-NEXT: class HelperC4 {};
// CHECK-NEW-CPP-NEXT: } // namespace
// CHECK-NEW-CPP-NEXT: } // namespace b

// CHECK-EMPTY: {{^}}{{$}}
