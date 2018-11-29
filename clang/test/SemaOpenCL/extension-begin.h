#ifndef INCLUDED
#define INCLUDED

#pragma OPENCL EXTENSION all : begin
#pragma OPENCL EXTENSION all : end

#pragma OPENCL EXTENSION my_ext : begin

struct A {
  int a;
};

typedef struct A TypedefOfA;
typedef const __private TypedefOfA* PointerOfA;

void f(void);

__attribute__((overloadable)) void g(long x);

#pragma OPENCL EXTENSION my_ext : end
#pragma OPENCL EXTENSION my_ext : end

__attribute__((overloadable)) void g(void);

#endif // INCLUDED

