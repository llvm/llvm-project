// RUN: %clang -### -target s390x-ibm-zos %s 2>&1 | FileCheck %s
// CHECK: SCEELIB(CELQS003)
// CHECK: libclang_rt.builtins.a
// RUN: %clang -### -target s390x-ibm-zos -nodefaultrtlibs %s 2>&1 | FileCheck %s -check-prefix=NORTLIB
// NORTLIB: SCEELIB(CELQS003)
// NORTLIB-NOT: libclang_rt.builtins.a
