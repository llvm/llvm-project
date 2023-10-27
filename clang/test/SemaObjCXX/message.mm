// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-fragile-10.5 -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-fragile-10.5 -verify -Wno-objc-root-class -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-fragile-10.5 -verify -Wno-objc-root-class -std=c++11 %s

@interface I1
- (int*)method;
@end

@implementation I1
- (int*)method {
  struct x { };
  [x method]; // expected-error{{receiver type 'x' is not an Objective-C class}}
  return 0;
}
@end

typedef struct { int x; } ivar;

@interface I2 {
  id ivar;
}
- (int*)method;
+ (void)method;
@end

struct I2_holder {
  I2_holder();

  I2 *get();
};

I2 *operator+(I2_holder, int);

@implementation I2
- (int*)method {
  [ivar method];

  // Test instance messages that start with a simple-type-specifier.
  [I2_holder().get() method];
  [I2_holder().get() + 17 method];
  return 0;
}
+ (void)method {
  [ivar method]; // expected-error{{receiver type 'ivar' is not an Objective-C class}}
}
@end

// Class message sends
@interface I3
+ (int*)method;
@end

@interface I4 : I3
+ (int*)otherMethod;
@end

template<typename T>
struct identity {
  typedef T type;
};

@implementation I4
+ (int *)otherMethod {
  // Test class messages that use non-trivial simple-type-specifiers
  // or typename-specifiers.
  if (false) {
    if (true)
      return [typename identity<I3>::type method];
#if __cplusplus <= 199711L
      // expected-warning@-2 {{'typename' occurs outside of a template}}
#endif

    return [::I3 method];
  }

  int* ip1 = {[super method]};
  int* ip2 = {[::I3 method]};
  int* ip3 = {[typename identity<I3>::type method]};
#if __cplusplus <= 199711L
  // expected-warning@-2 {{'typename' occurs outside of a template}}
#endif

  int* ip4 = {[typename identity<I2_holder>::type().get() method]};
#if __cplusplus <= 199711L
  // expected-warning@-2 {{'typename' occurs outside of a template}}
#endif
  int array[5] = {[3] = 2}; // expected-warning {{C99 extension}}
  return [super method];
}
@end

struct String {
  String(const char *);
};

struct MutableString : public String { };

// C++-specific parameter types
@interface I5
- method:(const String&)str1 
   other:(String&)str2; // expected-note{{passing argument to parameter 'str2' here}}
@end

void test_I5(I5 *i5, String s) {
  [i5 method:"hello" other:s];
  [i5 method:s other:"world"]; // expected-error{{non-const lvalue reference to type 'String' cannot bind to a value of unrelated type 'const char[6]'}}
}

@interface A

struct X { };

+ (A *)create:(void (*)(void *x, X r, void *data))callback
	      callbackData:(void *)callback_data;

@end


void foo(void)
{
  void *fun;
  void *ptr;
  X r;
  A *im = [A create:(void (*)(void *cgl_ctx, X r, void *data)) fun
             callbackData:ptr];
}

template<typename T> struct X1; // expected-note{{template is declared here}}

@interface B
+ (X1<int>)blah;
+ (X1<float>&)blarg;
@end

void f() {
  [B blah]; // expected-error{{implicit instantiation of undefined template 'X1<int>'}}
  [B blarg];
}
