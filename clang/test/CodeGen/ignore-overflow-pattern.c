// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-undefined-ignore-overflow-pattern=all %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-undefined-ignore-overflow-pattern=all -fwrapv %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-undefined-ignore-overflow-pattern=add-signed-overflow-test,add-unsigned-overflow-test %s -emit-llvm -o - | FileCheck %s --check-prefix=ADD
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-undefined-ignore-overflow-pattern=negated-unsigned-const %s -emit-llvm -o - | FileCheck %s --check-prefix=NEGATE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-undefined-ignore-overflow-pattern=unsigned-post-decr-while %s -emit-llvm -o - | FileCheck %s --check-prefix=WHILE

// Ensure some common overflow-dependent or overflow-prone code patterns don't
// trigger the overflow sanitizers. In many cases, overflow warnings caused by
// these patterns are seen as "noise" and result in users turning off
// sanitization all together.

// A pattern like "if (a + b < a)" simply checks for overflow and usually means
// the user is trying to handle it gracefully.

// Similarly, a pattern resembling "while (i--)" is extremely common and
// warning on its inevitable overflow can be seen as superfluous. Do note that
// using "i" in future calculations can be tricky because it will still
// wrap-around.

// Another common pattern that, in some cases, is found to be too noisy is
// unsigned negation, for example:
// unsigned long A = -1UL;

// Skip over parts of the IR containing this file's name.
// CHECK: source_filename = {{.*}}

// Ensure we don't see anything about handling overflow before the tests below.
// CHECK-NOT: handle{{.*}}overflow

extern unsigned a, b, c;
extern int u, v;
extern unsigned some(void);

// ADD-LABEL: @basic_commutativity
// WHILE-LABEL: @basic_commutativity
// NEGATE-LABEL: @basic_commutativity
// WHILE: handler.add_overflow
// NEGATE: handler.add_overflow
// ADD-NOT: handler.add_overflow
void basic_commutativity(void) {
  if (a + b < a)
    c = 9;
  if (a + b < b)
    c = 9;
  if (b + a < b)
    c = 9;
  if (b + a < a)
    c = 9;
  if (a > a + b)
    c = 9;
  if (a > b + a)
    c = 9;
  if (b > a + b)
    c = 9;
  if (b > b + a)
    c = 9;
  if (u + v < u)
    c = 9;
}

// ADD-LABEL: @arguments_and_commutativity
// WHILE-LABEL: @arguments_and_commutativity
// NEGATE-LABEL: @arguments_and_commutativity
// WHILE: handler.add_overflow
// NEGATE: handler.add_overflow
// ADD-NOT: handler.add_overflow
void arguments_and_commutativity(unsigned V1, unsigned V2) {
  if (V1 + V2 < V1)
    c = 9;
  if (V1 + V2 < V2)
    c = 9;
  if (V2 + V1 < V2)
    c = 9;
  if (V2 + V1 < V1)
    c = 9;
  if (V1 > V1 + V2)
    c = 9;
  if (V1 > V2 + V1)
    c = 9;
  if (V2 > V1 + V2)
    c = 9;
  if (V2 > V2 + V1)
    c = 9;
}

// ADD-LABEL: @pointers
// WHILE-LABEL: @pointers
// NEGATE-LABEL: @pointers
// WHILE: handler.add_overflow
// NEGATE: handler.add_overflow
// ADD-NOT: handler.add_overflow
void pointers(unsigned *P1, unsigned *P2, unsigned V1) {
  if (*P1 + *P2 < *P1)
    c = 9;
  if (*P1 + V1 < V1)
    c = 9;
  if (V1 + *P2 < *P2)
    c = 9;
}

struct OtherStruct {
  unsigned foo, bar;
};

struct MyStruct {
  unsigned base, offset;
  struct OtherStruct os;
};

extern struct MyStruct ms;

// ADD-LABEL: @structs
// WHILE-LABEL: @structs
// NEGATE-LABEL: @structs
// WHILE: handler.add_overflow
// NEGATE: handler.add_overflow
// ADD-NOT: handler.add_overflow
void structs(void) {
  if (ms.base + ms.offset < ms.base)
    c = 9;
}

// ADD-LABEL: @nestedstructs
// WHILE-LABEL: @nestedstructs
// NEGATE-LABEL: @nestedstructs
// WHILE: handler.add_overflow
// NEGATE: handler.add_overflow
// ADD-NOT: handler.add_overflow
void nestedstructs(void) {
  if (ms.os.foo + ms.os.bar < ms.os.foo)
    c = 9;
}

// ADD-LABEL: @constants
// WHILE-LABEL: @constants
// NEGATE-LABEL: @constants
// WHILE: handler.add_overflow
// NEGATE: handler.add_overflow
// ADD-NOT: handler.add_overflow
// Normally, this would be folded into a simple call to the overflow handler
// and a store. Excluding this pattern results in just a store.
void constants(void) {
  unsigned base = 4294967295;
  unsigned offset = 1;
  if (base + offset < base)
    c = 9;
}
// ADD-LABEL: @common_while
// NEGATE-LABEL: @common_while
// WHILE-LABEL: @common_while
// ADD: usub.with.overflow
// NEGATE: usub.with.overflow
// WHILE:  %dec = add i32 %0, -1
void common_while(unsigned i) {
  // This post-decrement usually causes overflow sanitizers to trip on the very
  // last operation.
  while (i--) {
    some();
  }
}

// ADD-LABEL: @negation
// NEGATE-LABEL: @negation
// WHILE-LABEL @negation
// ADD: negate_overflow
// NEGATE-NOT: negate_overflow
// WHILE: negate_overflow
// Normally, these assignments would trip the unsigned overflow sanitizer.
void negation(void) {
#define SOME -1UL
  unsigned long A = -1UL;
  unsigned long B = -2UL;
  unsigned long C = -SOME;
  (void)A;(void)B;(void)C;
}


// ADD-LABEL: @function_call
// WHILE-LABEL: @function_call
// NEGATE-LABEL: @function_call
// WHILE: handler.add_overflow
// NEGATE: handler.add_overflow
// ADD-NOT: handler.add_overflow
void function_call(void) {
  if (b + some() < b)
    c = 9;
}
