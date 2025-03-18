// RUN: %clang_cc1 %s -triple amdgcn-amd-amdhsa -fsyntax-only -verify

#define _AS0 __attribute__((address_space(0)))
#define _AS1 __attribute__((address_space(1)))
#define _AS2 __attribute__((address_space(2)))
#define _AS3 __attribute__((address_space(3)))
#define _AS4 __attribute__((address_space(4)))
#define _AS5 __attribute__((address_space(5)))
#define _AS999 __attribute__((address_space(999)))

void *p1(void _AS1 *p) { return p; }
void *p2(void _AS2 *p) { return p; } // expected-error {{returning '_AS2 void *' from a function with result type 'void *' changes address space of pointer}}
void *p3(void _AS3 *p) { return p; }
void *p4(void _AS4 *p) { return p; }
void *p5(void _AS5 *p) { return p; }
void *pi(void _AS999 *p) { return p; } // expected-error {{returning '_AS999 void *' from a function with result type 'void *' changes address space of pointer}}
void *pc(void __attribute__((opencl_local)) *p) { return p; } // expected-error {{returning '__local void *' from a function with result type 'void *' changes address space of pointer}}
void _AS1 *r0(void _AS1 *p) { return p; }
void _AS1 *r1(void *p) { return p; } // expected-error {{returning 'void *' from a function with result type '_AS1 void *' changes address space of pointer}}
void _AS1 *r2(void *p) { return (void _AS1 *)p; }
