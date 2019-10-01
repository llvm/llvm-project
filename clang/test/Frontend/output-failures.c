// RUN: not %clang_cc1 -emit-llvm -o %t.doesnotexist/somename %s 2> %t
// RUN: FileCheck -check-prefix=OUTPUTFAIL -input-file=%t %s

// OUTPUTFAIL: error: unable to open output file '{{.*}}doesnotexist{{.}}somename': '{{[nN]}}o such file or directory'
