// Tests without serialization:
// RUN: %clang_cc1 -ast-dump -triple i386-pc-linux-gnu %s \
// RUN: | FileCheck %s --strict-whitespace --check-prefixes=CHECK,CHECK1
//
// RUN: %clang_cc1 -ast-dump -triple aarch64-unknown-linux-gnu %s \
// RUN: | FileCheck %s --strict-whitespace --check-prefixes=CHECK,CHECK1
//
// RUN: %clang_cc1 -ast-dump -triple i386-pc-linux-gnu -DFAST -mreassociate %s \
// RUN: | FileCheck %s --strict-whitespace --check-prefixes=CHECK,CHECK1
//
// RUN: %clang_cc1 -ast-dump -triple aarch64-unknown-linux-gnu -DFAST -mreassociate %s \
// RUN: | FileCheck %s --strict-whitespace --check-prefixes=CHECK,CHECK1
//
// RUN: %clang_cc1 -ast-dump -triple i386-pc-linux-gnu -DFAST -mreassociate %s \
// RUN: -fprotect-parens \
// RUN: | FileCheck %s --strict-whitespace --check-prefixes=CHECK,CHECK2
//
// RUN: %clang_cc1 -ast-dump -triple aarch64-unknown-linux-gnu -DFAST -mreassociate %s \
// RUN: -fprotect-parens \
// RUN: | FileCheck %s --strict-whitespace --check-prefixes=CHECK,CHECK2
//
// Tests with serialization:
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -include-pch %t -ast-dump-all /dev/null \
// RUN: | FileCheck %s --strict-whitespace
//
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -include-pch %t -ast-dump-all /dev/null \
// RUN: | FileCheck %s --strict-whitespace
//
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -DFAST -mreassociate %s \
// RUN: -emit-pch -o %t
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -include-pch %t -ast-dump-all /dev/null \
// RUN: | FileCheck %s --strict-whitespace --check-prefixes=CHECK,CHECK1
//
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -DFAST -mreassociate %s \
// RUN: -emit-pch -o %t
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -include-pch %t -ast-dump-all /dev/null \
// RUN: | FileCheck %s --strict-whitespace --check-prefixes=CHECK,CHECK1
//
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -DFAST -mreassociate %s \
// RUN:   -fprotect-parens -emit-pch -o %t
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -include-pch %t -ast-dump-all /dev/null -fprotect-parens\
// RUN: | FileCheck %s --strict-whitespace --check-prefixes=CHECK,CHECK2
//
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -DFAST -mreassociate %s \
// RUN:   -fprotect-parens -emit-pch -o %t
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -include-pch %t -ast-dump-all /dev/null -fprotect-parens\
// RUN: | FileCheck %s --strict-whitespace --check-prefixes=CHECK,CHECK2

int v;
int addit(float a, float b) {

  v = __arithmetic_fence(a + b);

  v = (a + b);

  return 0;
}
//CHECK:| `-CompoundStmt {{.*}}
//CHECK-NEXT:|   |-BinaryOperator {{.*}} 'int' '='
//CHECK-NEXT:|   | |-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'v' 'int'
//CHECK-NEXT:|   | `-ImplicitCastExpr {{.*}}
//CHECK-NEXT:|   |   `-CallExpr {{.*}} 'float'
//CHECK-NEXT:|   |     |-ImplicitCastExpr {{.*}}
//CHECK-NEXT:|   |     | `-DeclRefExpr {{.*}}' Function {{.*}} '__arithmetic_fence'{{.*}}
//CHECK1-NOT:|   |     | `-DeclRefExpr {{.*}}' Function{{.*}} '__arithmetic_fence' 'void ()'
//CHECK2:|   |     | `-DeclRefExpr {{.*}} Function{{.*}} '__arithmetic_fence' 'void ()'
