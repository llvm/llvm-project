// RUN: %clang_cc1 -triple x86_64-apple-darwin -O0 %s -emit-llvm -o - | FileCheck %s

// Make sure the call to foo is compiled as:
//  call float @foo()
// not
//  call float (...)* bitcast (float ()* @foo to float (...)*)( )

static float foo() { return 0.0; }
// CHECK: call float @foo
float bar() { return foo()*10.0;}
