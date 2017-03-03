// RUN: %clang_analyze_cc1 -fblocks -analyzer-checker=core,nullability -verify %s

void it_takes_two(int a, int b);
void function_pointer_arity_mismatch() {
  void(*fptr)() = it_takes_two;
  fptr(1); // no-crash expected-warning {{Function taking 2 arguments is called with fewer (1)}}
}

void block_arity_mismatch() {
  void(^b)() = ^(int a, int b) { };
  b(1);  // no-crash expected-warning {{Block taking 2 arguments is called with fewer (1)}}
}
