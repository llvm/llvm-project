// RUN: rm -rf %t && mkdir -p %t/a %t/b

// RUN: %clang_cc1 -triple x86_64-apple-macos11 -emit-pch %s -o %t/a/t1.pch
// RUN: %clang_cc1 -triple x86_64-apple-macos11 -emit-pch %s -o %t/b/t2.pch

// RUN: diff %t/a/t1.pch %t/b/t2.pch
