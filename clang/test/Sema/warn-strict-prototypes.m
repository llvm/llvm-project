// RUN: %clang_cc1 -fsyntax-only -Wstrict-prototypes -verify -fblocks %s

@interface Foo

@property (nonatomic, copy) void (^noProtoBlock)(); // expected-warning {{a block declaration without a prototype is deprecated}}
@property (nonatomic, copy) void (^block)(void); // no warning

- doStuff:(void (^)()) completionHandler; // expected-warning {{a block declaration without a prototype is deprecated}}
- doOtherStuff:(void (^)(void)) completionHandler; // no warning

@end

void foo() { // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
  void (^block)() = // expected-warning {{a block declaration without a prototype is deprecated}}
                    ^void(int arg) { // no warning
  };
  // FIXME: this should say "a block declaration" instead, but block literal
  // expressions do not track their full declarator information, so we don't
  // know it's a block when diagnosing.
  void (^block2)(void) = ^void() {
  };
  void (^block3)(void) = ^ { // no warning
  };
}

void (*(^(*(^block4)()) // expected-warning {{a block declaration without a prototype is deprecated}}
     ()) // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
     ()) // expected-warning {{a block declaration without a prototype is deprecated}}
     (); // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
