// RUN: %clang_cc1 -verify -fsyntax-only -fblocks %s

@interface TestAttrMallocOnMethods {}
- (id) test1 __attribute((malloc)); //  expected-warning {{functions returning a pointer type}}
- (int) test2 __attribute((malloc)); //  expected-warning {{functions returning a pointer type}}
@end

id bar(void) __attribute((malloc)); // no-warning

typedef void (^bptr)(void);
bptr baz(void) __attribute((malloc)); // no-warning

__attribute((malloc)) id (*f)(); //  expected-warning {{functions returning a pointer type}}
__attribute((malloc)) bptr (*g)(); //  expected-warning {{functions returning a pointer type}}
__attribute((malloc)) void *(^h)(); //  expected-warning {{functions returning a pointer type}}

