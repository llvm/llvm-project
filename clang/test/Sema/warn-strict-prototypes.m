// RUN: %clang_cc1 -fsyntax-only -Wstrict-prototypes -verify -fblocks %s

@interface Foo

@property (nonatomic, copy) void (^noProtoBlock)(); // expected-warning {{this block declaration is not a prototype}}
@property (nonatomic, copy) void (^block)(void); // no warning

- doStuff:(void (^)()) completionHandler; // expected-warning {{this block declaration is not a prototype}}
- doOtherStuff:(void (^)(void)) completionHandler; // no warning

@end

void foo() {
  void (^block)() = // expected-warning {{this block declaration is not a prototype}}
                    ^void(int arg) { // no warning
  };
  void (^block2)(void) = ^void() { // no warning
  };
  void (^block3)(void) = ^ { // no warning
  };
}

void (*(^(*(^block4)()) // expected-warning {{this block declaration is not a prototype}}
     ()) // expected-warning {{this function declaration is not a prototype}}
     ()) // expected-warning {{this block declaration is not a prototype}}
     (); // expected-warning {{this function declaration is not a prototype}}
