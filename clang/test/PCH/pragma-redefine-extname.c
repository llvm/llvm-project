/// Test this without pch.
// RUN: %clang_cc1 -triple=x86_64-unknown-linux -include %S/pragma-redefine-extname.h %s -verify -emit-llvm -o - | FileCheck %s

/// Test with pch.
// RUN: %clang_cc1 -triple=x86_64-unknown-linux -x c-header -emit-pch -o %t %S/pragma-redefine-extname.h
// RUN: %clang_cc1 -triple=x86_64-unknown-linux -include-pch %t %s -verify -emit-llvm -o - | FileCheck %s

/// Compile it a few times to check that the PCH file is deterministic.
// RUN: %clang_cc1 -triple=x86_64-unknown-linux -x c-header -emit-pch -o %t.cmp %S/pragma-redefine-extname.h
// RUN: diff %t %t.cmp >/dev/null
// RUN: %clang_cc1 -triple=x86_64-unknown-linux -x c-header -emit-pch -o %t.cmp %S/pragma-redefine-extname.h
// RUN: diff %t %t.cmp >/dev/null
// RUN: %clang_cc1 -triple=x86_64-unknown-linux -x c-header -emit-pch -o %t.cmp %S/pragma-redefine-extname.h
// RUN: diff %t %t.cmp >/dev/null
// RUN: %clang_cc1 -triple=x86_64-unknown-linux -x c-header -emit-pch -o %t.cmp %S/pragma-redefine-extname.h
// RUN: diff %t %t.cmp >/dev/null

// CHECK: define dso_local void @redeffunc2_ext
// CHECK: call void @redeffunc1_ext

/// Issue #186742: check that #pragma redefine_extname exports into PCHs even if the header contains no declaration of the symbol
void undecfunc1(void);
void undecfunc2(void) { undecfunc1(); }
static void undecfunc3(void) {} // expected-warning {{#pragma redefine_extname is applicable to external C declarations only; not applied to function 'undecfunc3'}}
