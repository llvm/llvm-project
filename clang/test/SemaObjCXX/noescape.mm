// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -std=c++1z %s

typedef void (^BlockTy)();

struct S {
  int i;
  void m();
};

void noescapeFunc0(id, __attribute__((noescape)) BlockTy);
void noescapeFunc1(id, [[clang::noescape]] BlockTy);
void noescapeFunc2(__attribute__((noescape)) int *); // expected-note {{previous declaration is here}}
void noescapeFunc3(__attribute__((noescape)) id);
void noescapeFunc4(__attribute__((noescape)) int &);
void noescapeFunc2(int *); // expected-error {{conflicting types for 'noescapeFunc2'}}

void invalidFunc0(int __attribute__((noescape))); // expected-warning {{'noescape' attribute only applies to pointer arguments}}
void invalidFunc1(int __attribute__((noescape(0)))); // expected-error {{'noescape' attribute takes no arguments}}
void invalidFunc2(int0 *__attribute__((noescape))); // expected-error {{use of undeclared identifier 'int0'; did you mean 'int'?}}
void invalidFunc3(__attribute__((noescape)) int (S::*Ty)); // expected-warning {{'noescape' attribute only applies to pointer arguments}}
void invalidFunc4(__attribute__((noescape)) void (S::*Ty)()); // expected-warning {{'noescape' attribute only applies to pointer arguments}}
int __attribute__((noescape)) g; // expected-warning {{'noescape' attribute only applies to parameters}}

struct S1 {
  virtual void m0(int *__attribute__((noescape))); // expected-note {{parameter of overridden method is annotated with __attribute__((noescape))}}
};

struct S2 : S1 {
  void m0(int *__attribute__((noescape))) override;
};

struct S3 : S1 {
  void m0(int *) override; // expected-warning {{parameter of overriding method should be annotated with __attribute__((noescape))}}
};

__attribute__((objc_root_class))
@interface C0
-(void) m0:(int*)__attribute__((noescape)) p; // expected-note {{parameter of overridden method is annotated with __attribute__((noescape))}}
@end

@implementation C0
-(void) m0:(int*)__attribute__((noescape)) p {}
@end

@interface C1 : C0
-(void) m0:(int*)__attribute__((noescape)) p;
@end

@implementation C1 : C0
-(void) m0:(int*)__attribute__((noescape)) p {}
@end

@interface C2 : C0
-(void) m0:(int*) p; // expected-warning {{parameter of overriding method should be annotated with __attribute__((noescape))}}
@end

@implementation C2 : C0
-(void) m0:(int*) p {}
@end

void func0(int *);
void (*fnptr0)(int *);
void (*fnptr1)(__attribute__((noescape)) int *);
template<void (*fn)(int*)> struct S4 {};
template<void (*fn)(int* __attribute__((noescape)))> struct S5 {};

#if __cplusplus < 201406
  // expected-note@-4 {{template parameter is declared here}}
  // expected-note@-4 {{template parameter is declared here}}
#endif

void test0() {
  fnptr0 = &func0;
  fnptr0 = &noescapeFunc2;
  fnptr1 = &func0; // expected-error {{assigning to 'void (*)(__attribute__((noescape)) int *)' from incompatible type 'void (*)(int *)'}}
  fnptr1 = &noescapeFunc2;
  S4<&func0> e0;
  S4<&noescapeFunc2> e1;
  S5<&func0> ne0;

#if __cplusplus < 201406
  // expected-error@-4 {{non-type template argument of type 'void (*)(__attribute__((noescape)) int *)' cannot be converted to a value of type 'void (*)(int *)'}}
  // expected-error@-4 {{non-type template argument of type 'void (*)(int *)' cannot be converted to a value of type 'void (*)(__attribute__((noescape)) int *)'}}
#else
  // expected-error@-6 {{value of type 'void (*)(int *)' is not implicitly convertible to 'void (*)(__attribute__((noescape)) int *)'}}
#endif

  S5<&noescapeFunc2> ne1;
}

@protocol NoescapeProt
-(void) m0:(int*)__attribute__((noescape)) p; // expected-note 2 {{parameter of overridden method is annotated with __attribute__((noescape))}}
+(void) m1:(int*)__attribute__((noescape)) p;
-(void) m1:(int*) p;
@end

__attribute__((objc_root_class))
@interface C3
-(void) m0:(int*) p;
+(void) m1:(int*)__attribute__((noescape)) p;
-(void) m1:(int*) p;
@end

@interface C3 () <NoescapeProt> // expected-note {{class extension conforms to protocol 'NoescapeProt' which defines method 'm0:'}}
@end

@implementation C3
-(void) m0:(int*) p { // expected-warning {{parameter of overriding method should be annotated with __attribute__((noescape))}}
}
+(void) m1:(int*)__attribute__((noescape)) p {
}
-(void) m1:(int*) p {
}
@end

__attribute__((objc_root_class))
@interface C4 <NoescapeProt>
-(void) m0:(int*) p; // expected-warning {{parameter of overriding method should be annotated with __attribute__((noescape))}}
@end

@implementation C4
-(void) m0:(int*) p {
}
+(void) m1:(int*)__attribute__((noescape)) p {
}
-(void) m1:(int*) p {
}
@end
