// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface Foo
- (int)  garf; // expected-note {{previous declaration is here}}
- (int) OK;
+ (int)  cgarf; // expected-note {{previous declaration is here}}
- (int)  InstMeth;
@end

@interface Foo()
- (void)  garf; // expected-error {{duplicate declaration of method 'garf'}}
+ (void)  cgarf; // expected-error {{duplicate declaration of method 'cgarf'}}
+ (int)  InstMeth;
- (int) OK;
@end
