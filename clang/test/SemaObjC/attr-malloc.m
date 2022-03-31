// RUN: %clang_cc1 -verify -fsyntax-only -fblocks %s

@interface TestAttrMallocOnMethods {}
- (id) test1 __attribute((malloc)); //  expected-warning {{attribute only applies to functions}}
- (int) test2 __attribute((malloc)); //  expected-warning {{attribute only applies to functions}}
@end

id bar(void) __attribute((malloc)); // no-warning

typedef void (^bptr)(void);
bptr baz(void) __attribute((malloc)); // no-warning

__attribute((malloc)) id (*f)(void); //  expected-warning {{attribute only applies to functions}}
__attribute((malloc)) bptr (*g)(void); //  expected-warning {{attribute only applies to functions}}
__attribute((malloc)) void *(^h)(void); //  expected-warning {{attribute only applies to functions}}

