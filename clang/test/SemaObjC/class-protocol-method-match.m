// RUN: %clang_cc1  -Woverriding-method-mismatch -fsyntax-only -verify -Wno-objc-root-class %s

@protocol Bar 
@required
- (bycopy id)bud; // expected-note {{previous declaration is here}}
- (unsigned char) baz; // expected-note {{previous declaration is here}}
- (char) ok;
- (void) also_ok;
@end

@protocol Bar1 
@required
- (unsigned char) baz; // expected-note {{previous declaration is here}}
- (unsigned char) also_ok; // expected-note {{previous declaration is here}}
- (void) ban : (int) arg, ...; // expected-note {{previous declaration is here}}
@end

@protocol Baz <Bar, Bar1>
- (void) bar : (unsigned char)arg; // expected-note {{previous declaration is here}}
- (void) ok;
- (char) bak; // expected-note {{previous declaration is here}}
@end

@interface Foo <Baz>
- (id)bud; // expected-warning {{conflicting distributed object modifiers on return type in declaration of 'bud'}}
- (void) baz; // expected-warning 2 {{conflicting return type in declaration of 'baz': 'unsigned char' vs 'void'}}
- (void) bar : (unsigned char*)arg; // expected-warning {{conflicting parameter types in declaration of 'bar:': 'unsigned char' vs 'unsigned char *'}}
- (void) ok;
- (void) also_ok; // expected-warning {{conflicting return type in declaration of 'also_ok': 'unsigned char' vs 'void'}}
- (void) still_ok;
- (void) ban : (int) arg; // expected-warning {{conflicting variadic declaration of method and its implementation}}
@end

@interface Foo()
- (void) bak;
@end

@implementation Foo
- (bycopy id)bud { return 0; }
- (void) baz {}
- (void) bar : (unsigned char*)arg {}
- (void) ok {}
- (void) also_ok {}
- (void) still_ok {}
- (void) ban : (int) arg {}
- (void) bak {} // expected-warning {{conflicting return type in declaration of 'bak': 'char' vs 'void'}}
@end
