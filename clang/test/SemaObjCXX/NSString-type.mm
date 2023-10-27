// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fblocks -fsyntax-only -verify %s

void test(id pid, Class pclass) {
  void (^block)(void) = @"help"; // expected-error {{cannot initialize a variable of type 'void (^)()' with an rvalue of type 'NSString *'}}
  void (^block1)(void) = pid;
  void (^block2)(void) = @"help"; // expected-error {{cannot initialize a variable of type 'void (^)()' with an rvalue of type 'NSString *'}}
  void (^block3)(void) = @"help"; // expected-error {{cannot initialize a variable of type 'void (^)()' with an rvalue of type 'NSString *'}}
}

