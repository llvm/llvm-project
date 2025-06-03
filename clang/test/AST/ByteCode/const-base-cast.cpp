// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm %s -o - -fexperimental-new-constant-interpreter | FileCheck %s


/// Slightly adapted to the version from test/CodeGenCXX/.

struct X { int x[12];};
struct A : X { char x, y, z; };
struct B { char y; };
struct C : A,B {};
unsigned char x = ((char*)(X*)(C*)0x1000) - (char*)0x1000;
// CHECK: @x = {{(dso_local )?}}global i8 0

unsigned char y = ((char*)(B*)(C*)0x1000) - (char*)0x1000;
// CHECK: @y = {{(dso_local )?}}global i8 51

unsigned char z = ((char*)(A*)(C*)0x1000) - (char*)0x1000;
// CHECK: @z = {{(dso_local )?}}global i8 0

