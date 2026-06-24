// RUN: %clang_cc1 -triple i386-unknown-unknown -ast-dump %s | FileCheck %s

// CHECK: TypeAliasDecl {{0x[a-z0-9]*}} {{.*}}T1 'void (int) __attribute__((regparm (2)))'
using T1 [[gnu::regparm(2)]] = void(int);
