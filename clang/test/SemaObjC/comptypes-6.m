// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

@interface Derived
@end

@interface Object @end

extern Object* foo(void);

static Derived *test(void)
{
   Derived *m = foo();   // expected-warning {{incompatible pointer types initializing 'Derived *' with an expression of type 'Object *'}}

   return m;
}

