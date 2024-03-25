// RUN: %clang_cc1 -verify -o - %s

__attribute__((objc_root_class))
@interface Root @end

// Test reference binding.

typedef struct {
  int f0;
  int f1;
} T;

@interface A : Root
@property (assign) T p0;
@property (assign) T& p1; 
@end

int f0(const T& t) {
  return t.f0;
}

int f1(A *a) {
  return f0(a.p0);
}

int f2(A *a) {
  return f0(a.p1);	
}

// PR7740
@class NSString;

void f3(id);
void f4(NSString &tmpstr) {
  f3(&tmpstr);
}

// PR7741
@protocol P1 @end
@protocol P2 @end
@protocol P3 @end
@interface foo<P1> {} @end
@interface bar : foo <P1, P2, P3> {} @end
typedef bar baz;

struct ToBar {
  operator bar&() const;
};

void f5(foo&);
void f5b(foo<P1>&);
void f5c(foo<P2>&);
void f5d(foo<P3>&);
void f6(baz* x) { 
  f5(*x); 
  f5b(*x); 
  f5c(*x); 
  f5d(*x);
  (void)((foo&)*x);
  f5(ToBar());
  f5b(ToBar());
  f5c(ToBar());
  f5d(ToBar());
  (void)((foo&)ToBar());
}

@interface B : Root @end
@implementation B {
  unsigned bf : 4; // expected-note {{declared here}}
}

- (void) foo {
  unsigned &i = bf; // expected-error {{non-const reference cannot bind to bit-field 'bf'}}
}
@end
