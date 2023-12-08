// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-linux-gnu

// PR15216
// Don't crash when taking computing the offset of structs with large arrays.
const unsigned long Size = (1l << 60);

struct Chunk1 {
  char padding[Size]; // expected-warning {{folded to constant}}
  char more_padding[1][Size]; // expected-warning {{folded to constant}}
  char data;
};

int test1 = __builtin_offsetof(struct Chunk1, data);

struct Chunk2 {
  char padding[Size][Size][Size];  // expected-error {{array is too large}}
  char data;
};

// FIXME: Remove this error when the constant evaluator learns to
// ignore bad types.
int test2 = __builtin_offsetof(struct Chunk2, data);  // expected-error{{initializer element is not a compile-time constant}} 
