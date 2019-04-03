// RUN: %clang_cc1 -analyzer-config-help 2>&1 | FileCheck %s
// CHECK: OVERVIEW: Clang Static Analyzer -analyzer-config Option List
//
// CHECK: USAGE: clang -cc1 [CLANG_OPTIONS] -analyzer-config <OPTION1=VALUE,OPTION2=VALUE,...>
//
// CHECK:      clang -cc1 [CLANG_OPTIONS] -analyzer-config OPTION1=VALUE, -analyzer-config OPTION2=VALUE, ...
//
// CHECK:      clang [CLANG_OPTIONS] -Xclang -analyzer-config -Xclang<OPTION1=VALUE,OPTION2=VALUE,...>
//
// CHECK:      clang [CLANG_OPTIONS] -Xclang -analyzer-config -Xclang OPTION1=VALUE, -Xclang -analyzer-config -Xclang OPTION2=VALUE, ...
//
//
// CHECK: OPTIONS:
//
// CHECK: aggressive-binary-operation-simplification
// CHECK:                   (bool) Whether SValBuilder should rearrange
// CHECK:                   comparisons and additive operations of symbolic
// CHECK:                   expressions which consist of a sum of a
// CHECK:                   symbol and a concrete integer into the format
// CHECK:                   where symbols are on the left-hand side
// CHECK:                   and the integer is on the right. This is
// CHECK:                   only done if both symbols and both concrete
// CHECK:                   integers are signed, greater than or equal
// CHECK:                   to the quarter of the minimum value of the
// CHECK:                   type and less than or equal to the quarter
// CHECK:                   of the maximum value of that type. A
// CHECK:                   + n
// CHECK:                   <OP> B + m becomes A - B <OP> m - n, where
// CHECK:                   A and B symbolic, n and m are integers.
// CHECK:                   <OP> is any of '==', '!=', '<', '<=', '>',
// CHECK:                   '>=', '+' or '-'. The rearrangement also
// CHECK:                   happens with '-' instead of '+' on either
// CHECK:                   or both side and also if any or both integers
// CHECK:                   are missing. (default: false)
