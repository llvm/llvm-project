// XFAIL: system-zos
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: touch %t/A.pcm
// RUN: not %clang_cc1 -std=c++20 -fmodule-file=A=%t/A.pcm -ast-dump %s 2> %t/A.stderr | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-STDERR %s < %t/A.stderr

// CHECK-STDERR: fatal error: file '{{.*}}A.pcm' is not a valid module file
// CHECK: VarDecl {{.*}} n 'int'

import A;
import NonExistent;

int n;
