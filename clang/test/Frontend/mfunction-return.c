// RUN: %clang_cc1 -mfunction-return=keep -triple x86_64-linux-gnu %s
// RUN: %clang_cc1 -mfunction-return=thunk-extern -triple x86_64-linux-gnu %s

// RUN: not %clang_cc1 -mfunction-return=thunk -triple x86_64-linux-gnu %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-THUNK %s
// RUN: not %clang_cc1 -mfunction-return=thunk-inline -triple x86_64-linux-gnu %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-INLINE %s
// RUN: not %clang_cc1 -mfunction-return=invalid -triple x86_64-linux-gnu %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-INVALID %s
// RUN: not %clang_cc1 -mfunction-return=thunk-extern -triple s390x-linux-gnu %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-TARGET %s
// RUN: not %clang_cc1 -mfunction-return=thunk-extern -mcmodel=large \
// RUN:   -triple x86_64-linux-gnu %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LARGE %s

// CHECK-THUNK: error: invalid value 'thunk' in '-mfunction-return=thunk'
// CHECK-INLINE: error: invalid value 'thunk-inline' in '-mfunction-return=thunk-inline'
// CHECK-INVALID: error: invalid value 'invalid' in '-mfunction-return=invalid'
// CHECK-TARGET: error: invalid argument '-mfunction-return=' not allowed with 's390x-unknown-linux-gnu'
// CHECK-LARGE: error: invalid argument '-mfunction-return=thunk-extern' not allowed with '-mcmodel=large'
