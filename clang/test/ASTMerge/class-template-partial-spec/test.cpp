// FIXME: Crashes after r357394
// XFAIL: *
// RUN: %clang_cc1 -emit-pch -std=c++1z -o %t.1.ast %S/Inputs/class-template-partial-spec1.cpp
// RUN: %clang_cc1 -emit-pch -std=c++1z -o %t.2.ast %S/Inputs/class-template-partial-spec2.cpp
// RUN: %clang_cc1 -std=c++1z -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s

static_assert(sizeof(**SingleSource.member) == sizeof(**SingleDest.member));
static_assert(sizeof(SecondDoubleSource.member) == sizeof(SecondDoubleDest.member));
static_assert(NumberSource.val == 42);
static_assert(sizeof(Z0Source.member) == sizeof(char));
static_assert(sizeof(Dst::Z0Dst.member) == sizeof(double));
static_assert(sizeof(One::Child1<double, One::Two::Three::Parent<double>>::member) == sizeof(double));

// CHECK: class-template-partial-spec2.cpp:21:32: warning: external variable 'X1' declared with incompatible types in different translation units ('TwoOptionTemplate<int, double>' vs. 'TwoOptionTemplate<int, float>')
// CHECK: class-template-partial-spec1.cpp:21:31: note: declared here with type 'TwoOptionTemplate<int, float>'

// CHECK: class-template-partial-spec2.cpp:24:29: warning: external variable 'X4' declared with incompatible types in different translation units ('TwoOptionTemplate<int, int>' vs. 'TwoOptionTemplate<float, float>')
// CHECK: class-template-partial-spec1.cpp:24:33: note: declared here with type 'TwoOptionTemplate<float, float>'

// CHECK: class-template-partial-spec1.cpp:38:8: warning: type 'IntTemplateSpec<5, void *>' has incompatible definitions in different translation units
// CHECK: class-template-partial-spec1.cpp:39:7: note: field 'member' has type 'int' here
// CHECK: class-template-partial-spec2.cpp:39:10: note: field 'member' has type 'double' here

// CHECK: class-template-partial-spec2.cpp:52:25: warning: external variable 'Y3' declared with incompatible types in different translation units ('IntTemplateSpec<2, int>' vs. 'IntTemplateSpec<3, int>')
// CHECK: class-template-partial-spec1.cpp:52:25: note: declared here with type 'IntTemplateSpec<3, int>'

// CHECK-NOT: static_assert
