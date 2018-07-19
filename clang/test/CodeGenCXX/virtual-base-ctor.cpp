// RUN: %clang_cc1 %s -emit-llvm -triple %itanium_abi_triple -o - -O2 | opt - -S -globalopt -o - | FileCheck %s

struct B;
extern B x;
char y;
typedef __typeof(sizeof(int)) size_t;
struct A { int a; A() { y = ((size_t)this - (size_t)&x) / sizeof(void*); } };
struct B : virtual A { void* x; };    
B x;

// CHECK: @y = local_unnamed_addr global i8 2
