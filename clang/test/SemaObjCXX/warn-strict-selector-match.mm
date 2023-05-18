// RUN: %clang_cc1  -Wstrict-selector-match -fsyntax-only -verify %s

@interface Base
- (id) meth1: (Base *)arg1; 	// expected-note {{using}}
- (id) window;	// expected-note {{using}}
@end

@interface Derived: Base
- (id) meth1: (Derived *)arg1;	// expected-note {{also found}}
- (Base *) window;	// expected-note {{also found}}
@end

void foo(void) {
  id r;

  [r meth1:r];	// expected-warning {{multiple methods named 'meth1:' found}}
  [r window]; 	// expected-warning {{multiple methods named 'window' found}}
}
