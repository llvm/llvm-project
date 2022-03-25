// RUN: %clang_cc1 -triple avr -target-cpu atmega328 -emit-llvm %s -o - \
// RUN:     | FileCheck %s --check-prefix AVR
// RUN: %clang_cc1 -triple avr -target-cpu attiny40 -emit-llvm %s -o - \
// RUN:     | FileCheck %s --check-prefix TINY

// NOTE: All arguments are passed via the stack for functions with variable arguments.
// AVR:      define {{.*}} i8 @foo0(i8 {{.*}}, i8 {{.*}}, ...)
// TINY:     define {{.*}} i8 @foo0(i8 {{.*}}, i8 {{.*}}, ...)
// AVR-NOT:  define {{.*}} i8 @foo0(i8 {{.*}} signext {{.*}}, i8 {{.*}} signext {{.*}}, ...)
// TINY-NOT: define {{.*}} i8 @foo0(i8 {{.*}} signext {{.*}}, i8 {{.*}} signext {{.*}}, ...)
char foo0(char a, char b, ...) {
  return a + b;
}

// NOTE: All arguments are passed via registers on both avr and avrtiny.
// AVR:  define {{.*}} i8 @foo1(i32 {{.*}}, i8 {{.*}} signext {{.*}})
// TINY: define {{.*}} i8 @foo1(i32 {{.*}}, i8 {{.*}} signext {{.*}})
char foo1(long a, char b) {
  return a + b;
}

// NOTE: The argument `char c` is passed via registers on avr, while via the stack on avrtiny.
//       The argument `char b` costs 2 registers, so there is no vacant register left for
//       `char c` on avrtiny.
// AVR:      define {{.*}} i8 @foo2(i32 {{.*}}, i8 {{.*}} signext {{.*}}, i8 {{.*}} signext {{.*}})
// TINY:     define {{.*}} i8 @foo2(i32 {{.*}}, i8 {{.*}} signext {{.*}}, i8 {{.*}})
// TINY-NOT: define {{.*}} i8 @foo2(i32 {{.*}}, i8 {{.*}} signext {{.*}}, i8 {{.*}} signext {{.*}})
char foo2(long a, char b, char c) {
  return a + b + c;
}

// NOTE: On avr, the argument `a` costs 16 registers and `b` costs 2 registers, so
//       `c` has to be passed via the stack.
// AVR:      define {{.*}} i8 @foo3({{.*}}, i8 {{.*}} signext {{.*}}, i8 {{.*}})
// AVR-NOT:  define {{.*}} i8 @foo3({{.*}}, i8 {{.*}} signext {{.*}}, i8 {{.*}} signext {{.*}})
// TINY:     define {{.*}} i8 @foo3({{.*}}, i8 {{.*}}, i8 {{.*}})
// TINY-NOT: define {{.*}} i8 @foo3({{.*}}, i8 {{.*}} signext {{.*}}, i8 {{.*}} signext {{.*}})
struct s15 {
  char arr[15];
};
char foo3(struct s15 a, char b, char c) {
  return a.arr[b] + a.arr[c];
}

// NOTE: On avr, `a` only costs 16 registers, though there are 2 vacant registers,
//       both `b` and `c` have to be passed via the stack.
// AVR:      define {{.*}} i8 @foo4({{.*}}, i32 {{.*}}, i8 {{.*}})
// AVR-NOT:  define {{.*}} i8 @foo4({{.*}}, i32 {{.*}}, i8 {{.*}} signext {{.*}})
// TINY:     define {{.*}} i8 @foo4({{.*}}, i32 {{.*}}, i8 {{.*}})
// TINY-NOT: define {{.*}} i8 @foo4({{.*}}, i32 {{.*}}, i8 {{.*}} signext {{.*}})
char foo4(struct s15 a, long b, char c) {
  return a.arr[c];
}

// NOTE: On avrtiny, `a` only costs 4 registers, though there are 2 vacant
//       registers, both `b` and `c` are passed via the stack.
// AVR:      define {{.*}} i8 @foo5(i32 {{.*}}, i32 {{.*}}, i8 {{.*}} signext {{.*}})
// TINY:     define {{.*}} i8 @foo5(i32 {{.*}}, i32 {{.*}}, i8 {{.*}})
// TINY-NOT: define {{.*}} i8 @foo5(i32 {{.*}}, i32 {{.*}}, i8 {{.*}} signext {{.*}})
char foo5(long a, long b, char c) {
  return c + 1;
}

// NOTE: All arguments are passed via the stack, though all registers are vacant.
// AVR:      define {{.*}} i8 @foo6({{.*}}, i8 {{.*}})
// AVR-NOT:  define {{.*}} i8 @foo6({{.*}}, i8 {{.*}} signext {{.*}})
// TINY:     define {{.*}} i8 @foo6({{.*}}, i8 {{.*}})
// TINY-NOT: define {{.*}} i8 @foo6({{.*}}, i8 {{.*}} signext {{.*}})
struct s32 {
  char arr[32];
};
char foo6(struct s32 a, char b) {
  return a.arr[b];
}

// NOTE: All arguments are passed via registers on avr. While all arguments are passed
//       via the stack on avrtiny, though all registers are vacant.
// AVR:      define {{.*}} i8 @foo7({{.*}}, i8 {{.*}} signext {{.*}})
// TINY:     define {{.*}} i8 @foo7({{.*}}, i8 {{.*}})
// TINY-NOT: define {{.*}} i8 @foo7({{.*}}, i8 {{.*}} signext {{.*}})
char foo7(struct s15 a, char b) {
  return a.arr[b];
}

// NOTE: On avr, though `a` only cost 16 registers, `b` has to be passed via the
//       stack, since there is an implicit pointer argument costs 2 registers.
// AVR:      define {{.*}} @foo8({{.*}}, {{.*}}, i8 {{.*}})
// AVR-NOT:  define {{.*}} @foo8({{.*}}, {{.*}}, i8 {{.*}} signext {{.*}})
// TINY:     define {{.*}} @foo8({{.*}}, {{.*}}, i8 {{.*}})
// TINY-NOT: define {{.*}} @foo8({{.*}}, {{.*}}, i8 {{.*}} signext {{.*}})
struct s15 foo8(struct s15 a, char b) {
  a.arr[0] = b;
  return a;
}

// NOTE: On avrtiny, `b` has to be passed via the stack, since there is an
//       implicit pointer argument costs 2 registers.
// AVR:      define {{.*}} @foo9({{.*}}, i32 {{.*}}, i8 {{.*}} signext {{.*}})
// TINY:     define {{.*}} @foo9({{.*}}, i32 {{.*}}, i8 {{.*}})
// TINY-NOT: define {{.*}} @foo9({{.*}}, i32 {{.*}}, i8 {{.*}} signext {{.*}})
struct s15 foo9(long a, char b) {
  struct s15 x;
  x.arr[0] = b;
  return x;
}

// NOTE: All arguments are passed via registers, though there is an implicit
//       pointer argument costs 2 registers.
// AVR:  define {{.*}} @fooa({{.*}}, i8 {{.*}} signext {{.*}}, i8 {{.*}} signext {{.*}})
// TINY: define {{.*}} @fooa({{.*}}, i8 {{.*}} signext {{.*}}, i8 {{.*}} signext {{.*}})
struct s15 fooa(char a, char b) {
  struct s15 x;
  x.arr[0] = a;
  x.arr[1] = b;
  return x;
}
