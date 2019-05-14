// RUN: %clang_cc1 -std=c++1z -emit-pch -o %t.1.ast %S/Inputs/class-template1.cpp
// RUN: %clang_cc1 -std=c++1z -emit-pch -o %t.2.ast %S/Inputs/class-template2.cpp
// RUN: %clang_cc1 -std=c++1z  -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s

static_assert(sizeof(X0<char>().getValue(1)) == sizeof(char));
static_assert(sizeof(X0<int>().getValue(1)) == sizeof(int));

// CHECK: class-template1.cpp:9:14: warning: non-type template parameter declared with incompatible types in different translation units ('int' vs. 'long')
// CHECK: class-template2.cpp:9:15: note: declared here with type 'long'

// CHECK: class-template1.cpp:12:14: warning: template parameter has different kinds in different translation units
// CHECK: class-template2.cpp:12:10: note: template parameter declared here

// CHECK: class-template1.cpp:18:23: warning: non-type template parameter declared with incompatible types in different translation units ('long' vs. 'int')
// CHECK: class-template2.cpp:18:23: note: declared here with type 'int'

// CHECK: class-template1.cpp:21:10: warning: template parameter has different kinds in different translation units
// CHECK: class-template2.cpp:21:10: note: template parameter declared here

// CHECK: class-template2.cpp:27:20: warning: external variable 'x0r' declared with incompatible types in different translation units ('X0<double> *' vs. 'X0<float> *')
// CHECK: class-template1.cpp:26:19: note: declared here with type 'X0<float> *'

// CHECK: class-template1.cpp:35:8: warning: type 'X0<wchar_t>' has incompatible definitions in different translation units
// CHECK: class-template1.cpp:36:7: note: field 'member' has type 'int' here
// CHECK: class-template2.cpp:36:9: note: field 'member' has type 'float' here

// CHECK: 6 warnings generated.
// CHECK-NOT: static_assert
