// RUN: %clang_cc1  -triple x86_64-apple-darwin10 -fblocks -fsyntax-only -verify %s
// rdar://10907410

void test(id pid, Class pclass) {
  void (^block)(void) = @"help"; // expected-error {{initializing 'void (^)(void)' with an expression of incompatible type 'NSString *'}}
  void (^block1)(void) = pid;
  void (^block2)(void) = @"help"; // expected-error {{initializing 'void (^)(void)' with an expression of incompatible type 'NSString *'}}
  void (^block3)(void) = @"help"; // expected-error {{initializing 'void (^)(void)' with an expression of incompatible type 'NSString *'}}
}

