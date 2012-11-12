// RUN: %clang_cc1 %s -fsyntax-only -verify -triple=x86_64-apple-darwin10

#define CHECK_SIZE(name, size) extern int name##1[sizeof(name) == size ? 1 : -1];
#define CHECK_ALIGN(name, size) extern int name##2[__alignof(name) == size ? 1 : -1];

// Simple tests.
struct Test1 {
  char c : 9; // expected-warning {{size of bit-field 'c' (9 bits) exceeds the size of its type; value will be truncated to 8 bits}}
};
CHECK_SIZE(Test1, 2);
CHECK_ALIGN(Test1, 1);

struct Test2 {
  char c : 16; // expected-warning {{size of bit-field 'c' (16 bits) exceeds the size of its type; value will be truncated to 8 bits}}
};
CHECK_SIZE(Test2, 2);
CHECK_ALIGN(Test2, 2);

struct Test3 {
  char c : 32; // expected-warning {{size of bit-field 'c' (32 bits) exceeds the size of its type; value will be truncated to 8 bits}}
};
CHECK_SIZE(Test3, 4);
CHECK_ALIGN(Test3, 4);

struct Test4 {
  char c : 64; // expected-warning {{size of bit-field 'c' (64 bits) exceeds the size of its type; value will be truncated to 8 bits}}
};
CHECK_SIZE(Test4, 8);
CHECK_ALIGN(Test4, 8);

