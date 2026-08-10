// RUN: %clang_cc1 %s -verify -Wobjc-signed-char-bool
// RUN: %clang_cc1 -xobjective-c++ %s -verify -Wobjc-signed-char-bool

typedef signed char BOOL;
#define YES __objc_yes
#define NO __objc_no

typedef unsigned char Boolean;

BOOL b;
Boolean boolean;
float fl;
int i;
int *ptr;

void t1(void) {
  b = boolean;
  b = fl; // expected-warning {{implicit conversion from floating-point type 'float' to 'BOOL'}}
  b = i; // expected-warning {{implicit conversion from integral type 'int' to 'BOOL'}}

  b = 1.0;
  b = 0.0;
  b = 1.1; // expected-warning {{implicit conversion from 'double' to 'BOOL' (aka 'signed char') changes value from 1.1 to 1}}
  b = 2.1; // expected-warning {{implicit conversion from constant value 2.1 to 'BOOL'; the only well defined values for 'BOOL' are YES and NO}}

  b = YES;
  b = ptr; // expected-error {{incompatible pointer to integer conversion assigning to 'BOOL' (aka 'signed char') from 'int *'}}
}

@interface BoolProp
@property BOOL p;
@end

void t2(BoolProp *bp) {
  bp.p = YES;
  bp.p = NO;
  bp.p = boolean;
  bp.p = fl; // expected-warning {{implicit conversion from floating-point type 'float' to 'BOOL'}}
  bp.p = i; // expected-warning {{implicit conversion from integral type 'int' to 'BOOL'}}
  bp.p = b;
  bp.p = bp.p;
  bp.p = ptr; // expected-error {{incompatible pointer to integer conversion assigning to 'BOOL' (aka 'signed char') from 'int *'}}
  bp.p = 1;
  bp.p = 2; // expected-warning {{implicit conversion from constant value 2 to 'BOOL'; the only well defined values for 'BOOL' are YES and NO}}
}

struct has_bf {
  int signed_bf1 : 1;
  int signed_bf2 : 2;
  unsigned unsigned_bf1 : 1;
  unsigned unsigned_bf2 : 2;

  struct has_bf *nested;
};

void t3(struct has_bf *bf) {
  b = bf->signed_bf1; // expected-warning{{implicit conversion from integral type 'int' to 'BOOL'}}
  b = bf->signed_bf2; // expected-warning{{implicit conversion from integral type 'int' to 'BOOL'}}
  b = bf->unsigned_bf1; // no warning
  b = bf->unsigned_bf2; // expected-warning{{implicit conversion from integral type 'unsigned int' to 'BOOL'}}
  struct has_bf local;
  b = local.unsigned_bf1;
  b = local.unsigned_bf2; // expected-warning{{implicit conversion from integral type 'unsigned int' to 'BOOL'}}
  b = local.nested->unsigned_bf1;
  b = local.nested->unsigned_bf2; // expected-warning{{implicit conversion from integral type 'unsigned int' to 'BOOL'}}
}

void t4(BoolProp *bp) {
  BOOL local = YES;
  bp.p = 1 ? local : NO; // no warning
}

__attribute__((objc_root_class))
@interface BFIvar {
  struct has_bf bf;
  unsigned unsigned_bf1 : 1;
  unsigned unsigned_bf2 : 2;
}
@end

@implementation BFIvar
-(void)m {
  b = bf.unsigned_bf1;
  b = bf.unsigned_bf2; // expected-warning{{implicit conversion from integral type 'unsigned int' to 'BOOL'}}
  b = unsigned_bf1;
  b = unsigned_bf2; // expected-warning{{implicit conversion from integral type 'unsigned int' to 'BOOL'}}
}
@end

#ifdef __cplusplus
template <class T>
struct S {
  unsigned i : sizeof(T);
};

template <class T>
void f() {
  S<T> i;
  BOOL b = i.i; // expected-warning{{implicit conversion from integral type 'unsigned int' to 'BOOL'}}
}

int main() {
  f<char>();
  f<short>(); // expected-note {{in instantiation of function template specialization 'f<short>' requested here}}
}
#endif

void t5(BOOL b) {
  int i;
  b = b ?: YES; // no warning
  b = b ?: i; // expected-warning {{implicit conversion from integral type 'int' to 'BOOL'}}
  b = (b = i) // expected-warning {{implicit conversion from integral type 'int' to 'BOOL'}}
               ?: YES;
  b = (1 ? YES : i) ?: YES; // expected-warning {{implicit conversion from integral type 'int' to 'BOOL'}}
  b = b ?: (1 ? i : i); // expected-warning 2 {{implicit conversion from integral type 'int' to 'BOOL'}}

  b = b ? YES : (i ?: 0); // expected-warning {{implicit conversion from integral type 'int' to 'BOOL'}}
}
