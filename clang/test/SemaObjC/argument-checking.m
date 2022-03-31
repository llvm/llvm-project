// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

struct S { int a; };

extern int charStarFunc(char *); // expected-note{{passing argument to parameter here}}
extern int charFunc(char); // expected-note{{passing argument to parameter here}}

@interface Test
+alloc;
-(int)charStarMeth:(char *)s; // expected-note{{passing argument to parameter 's' here}}
-structMeth:(struct S)s; // expected-note{{passing argument to parameter 's' here}}
-structMeth:(struct S)s 
   :(struct S)s2; // expected-note{{passing argument to parameter 's2' here}}
@end

void test(void) {
  id obj = [Test alloc];
  struct S sInst;

  charStarFunc(1); // expected-warning {{incompatible integer to pointer conversion passing 'int' to parameter of type 'char *'}}
  charFunc("abc"); // expected-warning {{incompatible pointer to integer conversion passing 'char[4]' to parameter of type 'char'}}

  [obj charStarMeth:1]; // expected-warning {{incompatible integer to pointer conversion sending 'int'}}
  [obj structMeth:1]; // expected-error {{sending 'int'}}
  [obj structMeth:sInst :1]; // expected-error {{sending 'int'}}
}
