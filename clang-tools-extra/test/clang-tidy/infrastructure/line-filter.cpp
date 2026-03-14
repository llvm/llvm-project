// RUN: clang-tidy -checks='-*,google-explicit-constructor' -line-filter='[{"name":"line-filter.cpp","lines":[[18,18],[22,22]]},{"name":"header1.h","lines":[[1,2]]},{"name":"header2.h"},{"name":"header3.h"}]' -header-filter='header[12]\.h$' %s -- -I %S/Inputs/line-filter 2>&1 | FileCheck %s

#include "header1.h"
// CHECK-NOT: header1.h:{{.*}} warning
// CHECK: header1.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK: header1.h:2:12: warning: single-argument constructors {{.*}}
// CHECK-NOT: header1.h:{{.*}} warning

#include "header2.h"
// CHECK: header2.h:1:12: warning: single-argument constructors {{.*}}
// CHECK: header2.h:2:12: warning: single-argument constructors {{.*}}
// CHECK: header2.h:3:12: warning: single-argument constructors {{.*}}
// CHECK-NOT: header2.h:{{.*}} warning

#include "header3.h"
// CHECK-NOT: header3.h:{{.*}} warning

class A { A(int); };
// CHECK: :[[@LINE-1]]:11: warning: single-argument constructors {{.*}}
class B { B(int); };
// CHECK-NOT: :[[@LINE-1]]:{{.*}} warning
class C { C(int); };
// CHECK: :[[@LINE-1]]:11: warning: single-argument constructors {{.*}}

// CHECK-NOT: warning:

// CHECK: Suppressed 3 warnings (1 in non-user code, 2 due to line filter)
