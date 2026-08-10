// RUN: %clang_cc1 -O2 -emit-llvm %s -o - | grep call

static int bar(int x, int y) __attribute__((noinline));

static int bar(int x, int y)  
{
 return x + y;
}

int foo(int a, int b) {
 return  bar(b, a);
}

