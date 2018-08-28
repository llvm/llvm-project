// RUN: %clang_cc1 -debug-info-kind=limited -triple armv6-apple-darwin9 -emit-llvm -o %t %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple armv6-unknown-unknown -emit-llvm -o %t %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple i686-apple-darwin9 -emit-llvm -o %t %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple i686-pc-linux-gnu -emit-llvm -o %t %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple i686-unknown-dragonfly -emit-llvm -o %t %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple i686-unknown-unknown -emit-llvm -o %t %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple i686-unknown-win32 -emit-llvm -o %t %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple powerpc-unknown-unknown -emit-llvm -o %t %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple powerpc64-unknown-unknown -emit-llvm -o %t %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple sparc-unknown-solaris -emit-llvm -o %t %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple sparc-unknown-unknown -emit-llvm -o %t %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple x86_64-apple-darwin9 -emit-llvm -o %t %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple x86_64-pc-linux-gnu -emit-llvm -o %t %s
// RUN: %clang_cc1 -debug-info-kind=limited -triple x86_64-unknown-unknown -emit-llvm -o %t %s

// <rdar://problem/7181838> clang 1.0 fails to compile Python 2.6
// RUN: %clang -target x86_64-apple-darwin9 -### -S %s -mmacosx-version-min=10.4
