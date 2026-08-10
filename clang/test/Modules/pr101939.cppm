// RUN: %clang_cc1 -std=c++20 %s -ast-dump | FileCheck %s

export module mod;
export auto a = __builtin_expect(true, true);

// CHECK-NOT: FunctionDecl{{.*}} in mod {{.*}} __builtin_expect
