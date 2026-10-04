// RUN: clang-tidy -checks='-*,google-explicit-constructor' --config='{}' -header-filter='^(?!.*/system/).*' %s -- -I %S/Inputs/file-filter 2>&1 | FileCheck --check-prefix=CHECK-HEADER %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' --config='{}' -header-filter='.*' -exclude-header-filter='(' %s -- -I %S/Inputs/file-filter 2>&1 | FileCheck --check-prefix=CHECK-EXCLUDE %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' --config='{HeaderFilterRegex: "(?!x)"}' %s -- -I %S/Inputs/file-filter 2>&1 | FileCheck --check-prefix=CHECK-CONFIG %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' --config='{}' -header-filter='' -exclude-header-filter='' %s -- -I %S/Inputs/file-filter 2>&1 | FileCheck --check-prefix=CHECK-EMPTY -implicit-check-not='clang-tidy-config' %s

// CHECK-HEADER: warning: Invalid header filter regex '^(?!.*/system/).*': repetition-operator operand invalid [clang-tidy-config]
// CHECK-EXCLUDE: warning: Invalid exclude header filter regex '(': parentheses not balanced [clang-tidy-config]
// CHECK-CONFIG: warning: Invalid header filter regex '(?!x)': repetition-operator operand invalid [clang-tidy-config]

#include "header1.h"
// An invalid header filter still matches no headers, and an invalid exclude
// header filter still excludes none.
// CHECK-HEADER-NOT: header1.h:1:12: warning:
// CHECK-EXCLUDE: header1.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK-CONFIG-NOT: header1.h:1:12: warning:
// CHECK-EMPTY-NOT: header1.h:1:12: warning:

class A { A(int); };
// CHECK-HEADER: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit
// CHECK-EXCLUDE: :[[@LINE-2]]:11: warning: single-argument constructors must be marked explicit
// CHECK-CONFIG: :[[@LINE-3]]:11: warning: single-argument constructors must be marked explicit
// CHECK-EMPTY: :[[@LINE-4]]:11: warning: single-argument constructors must be marked explicit
