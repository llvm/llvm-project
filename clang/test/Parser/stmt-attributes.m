// RUN: %clang_cc1 -verify %s \
// RUN:   -fblocks -fobjc-exceptions -fexceptions -fsyntax-only \
// RUN:   -Wno-unused-value -Wno-unused-getter-return-value

#if !__has_extension(statement_attributes_with_gnu_syntax)
#error "We should have statement attributes with GNU syntax support"
#endif

@interface Base
@end

@interface Test : Base
@property(getter=hasFoobar) int foobar;
- (void)foo;
- (void)bar;
@end

Test *getTest(void);

@implementation Test
- (void)foo __attribute__((nomerge)) {
  // expected-error@-1 {{'nomerge' attribute only applies to functions, statements and variables}}
}

- (void)bar {
  __attribute__(()) [self foo];
  // expected-error@-1 {{missing '[' at start of message send expression}}
  // expected-error@-2 {{expected ']'}}
  // expected-error@-3 {{expected identifier or '('}}
  // expected-note@-4 {{to match this '['}}
  __attribute__((nomerge)) [self foo];
  // expected-warning@-1 {{'nomerge' attribute is ignored because there exists no call expression inside the statement}}
  __attribute__((nomerge)) [getTest() foo];

  __attribute__(()) ^{};
  // expected-error@-1 {{expected identifier or '('}}
  __attribute__((nomerge)) ^{};
  // expected-warning@-1 {{'nomerge' attribute is ignored because there exists no call expression inside the statement}}
  __attribute__((nomerge)) ^{ [self foo]; }();

  __attribute__(()) @try {
    [self foo];
  } @finally {
  }

  __attribute__((nomerge)) @try {
    [getTest() foo];
  } @finally {
  }

  __attribute__((nomerge)) (__bridge void *)self;
  // expected-warning@-1 {{'nomerge' attribute is ignored because there exists no call expression inside the statement}}

  __attribute__((nomerge)) self.hasFoobar;
  // expected-warning@-1 {{'nomerge' attribute is ignored because there exists no call expression inside the statement}}
}
@end
