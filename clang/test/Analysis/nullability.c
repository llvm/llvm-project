// RUN: %clang_cc1 -fblocks -analyze -analyzer-checker=core,nullability -verify %s

void it_takes_two(int a, int b);
void function_pointer_arity_mismatch() {
  void(*fptr)() = it_takes_two;
  fptr(1); // no-crash expected-warning {{Function taking 2 arguments is called with less (1)}}
}

void block_arity_mismatch() {
  void(^b)() = ^(int a, int b) { }; // no-crash
  b(1);
}
