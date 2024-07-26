// Ensure some common idioms don't trigger the overflow sanitizers when
// -fno-sanitize-overflow-idioms is enabled. In many cases, overflow warnings
// caused by these idioms are seen as "noise" and result in users turning off
// sanitization all together.

// A pattern like "if (a + b < a)" simply checks for overflow and usually means
// the user is trying to handle it gracefully.

// Similarly, a pattern resembling "while (i--)" is extremely common and
// warning on its inevitable overflow can be seen as superfluous. Do note that
// using "i" in future calculations can be tricky because it will still
// wrap-around. Using -fno-sanitize-overflow-idioms or not doesn't change this
// fact -- we just won't warn/trap with sanitizers.

// Another common pattern that, in some cases, is found to be too noisy is
// unsigned negation, for example:
// unsigned long A = -1UL;

// RUN: %clang %s -O2 -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-overflow-pattern-exclusion=all -S -emit-llvm -o - | FileCheck %s
// RUN: %clang %s -O2 -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-overflow-pattern-exclusion=all -fwrapv -S -emit-llvm -o - | FileCheck %s
// RUN: %clang %s -O2 -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-overflow-pattern-exclusion=add-overflow-test -S -emit-llvm -o - | FileCheck %s --check-prefix=ADD
// RUN: %clang %s -O2 -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-overflow-pattern-exclusion=negated-unsigned-const -S -emit-llvm -o - | FileCheck %s --check-prefix=NEGATE
// RUN: %clang %s -O2 -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-overflow-pattern-exclusion=post-decr-while -S -emit-llvm -o - | FileCheck %s --check-prefix=WHILE

// CHECK-NOT: handle{{.*}}overflow

// ADD: usub.with.overflow
// ADD: negate_overflow
// ADD-NOT: handler.add_overflow

// NEGATE: handler.add_overflow
// NEGATE: usub.with.overflow
// NEGATE-NOT: negate_overflow

// WHILE: handler.add_overflow
// WHILE: negate_overflow
// WHILE-NOT: usub.with.overflow
extern unsigned a, b, c;
extern unsigned some(void);

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
}

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

void structs(void) {
  if (ms.base + ms.offset < ms.base)
    c = 9;
}

void nestedstructs(void) {
  if (ms.os.foo + ms.os.bar < ms.os.foo)
    c = 9;
}

// Normally, this would be folded into a simple call to the overflow handler
// and a store. Excluding this idiom results in just a store.
void constants(void) {
  unsigned base = 4294967295;
  unsigned offset = 1;
  if (base + offset < base)
    c = 9;
}

void common_while(unsigned i) {
  // This post-decrement usually causes overflow sanitizers to trip on the very
  // last operation.
  while (i--) {
    some();
  }
}

// Normally, these assignments would trip the unsigned overflow sanitizer.
void negation(void) {
#define SOME -1UL
  unsigned long A = -1UL;
  unsigned long B = -2UL;
  unsigned long C = -3UL;
  unsigned long D = -SOME;
  (void)A;(void)B;(void)C;(void)D;
}

// cvise'd kernel code that caused problems during development due to sign
// extension
typedef unsigned long size_t;
int qnbytes;
int *key_alloc_key;
size_t key_alloc_quotalen;
int *key_alloc(void) {
  if (qnbytes + key_alloc_quotalen < qnbytes)
    return key_alloc_key;
  return key_alloc_key + 3;;
}

void function_call(void) {
  if (b + some() < b)
    c = 9;
}
