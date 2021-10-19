#ifndef INCLUDED
#define INCLUDED

#pragma OPENCL EXTENSION all : begin
#pragma OPENCL EXTENSION all : end

#pragma OPENCL EXTENSION my_ext : begin
struct A {
  int a;
};
#pragma OPENCL EXTENSION my_ext : end
#pragma OPENCL EXTENSION my_ext : end

#define my_ext

typedef struct A TypedefOfA;
typedef const __private TypedefOfA* PointerOfA;

void f(void);

__attribute__((overloadable)) void g(long x);



__attribute__((overloadable)) void g(void);

#endif // INCLUDED
