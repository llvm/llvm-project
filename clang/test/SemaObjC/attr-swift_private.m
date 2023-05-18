// RUN: %clang_cc1 -verify -fsyntax-only -fobjc-arc %s

__attribute__((__swift_private__))
@protocol P
@end

__attribute__((__swift_private__))
@interface I
@end

@interface J
@property id property __attribute__((__swift_private__));
- (void)instanceMethod __attribute__((__swift_private__));
+ (void)classMethod __attribute__((__swift_private__));
@end

void f(void) __attribute__((__swift_private__));

struct __attribute__((__swift_private__)) S {};

enum __attribute__((__swift_private__)) E {
  one,
  two,
};

typedef struct { } T __attribute__((__swift_private__));

void g(void) __attribute__((__swift_private__("private")));
// expected-error@-1 {{'__swift_private__' attribute takes no arguments}}
