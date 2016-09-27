// RUN: %clang_cc1 -emit-llvm -o /dev/null -stats-file=%t %s
// RUN: FileCheck -input-file=%t %s
// CHECK: {
//  ... here come some json values ...
// CHECK: }

// RUN: %clang_cc1 -emit-llvm -o %t -stats-file=%S/doesnotexist/bla %s 2>&1 | FileCheck -check-prefix=OUTPUTFAIL %s
// OUTPUTFAIL: warning: unable to open statistics output file '{{.*}}doesnotexist{{.}}bla': '{{[Nn]}}o such file or directory'
