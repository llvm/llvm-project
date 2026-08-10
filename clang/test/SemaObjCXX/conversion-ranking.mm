// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
@protocol P1
@end

@interface A <P1>
@end

@interface B : A
@end

@interface C : B
@end

template<typename T>
struct ConvertsTo {
  operator T() const;
};


// conversion of C* to B* is better than conversion of C* to A*.
int &f0(A*);
float &f0(B*);

void test_f0(C *c) {
  float &fr1 = f0(c);
}

// conversion of B* to A* is better than conversion of C* to A*
void f1(A*);

struct ConvertsToBoth {
private:
  operator C*() const;

public:
  operator B*() const;
};

void test_f1(ConvertsTo<B*> toB, ConvertsTo<C*> toC, ConvertsToBoth toBoth) {
  f1(toB);
  f1(toC);
  f1(toBoth);
};

// A conversion to an a non-id object pointer type is better than a 
// conversion to 'id'.
int &f2(A*);
float &f2(id);

void test_f2(B *b) {
  int &ir = f2(b);
}

// A conversion to an a non-Class object pointer type is better than a 
// conversion to 'Class'.
int &f3(A*);
float &f3(Class);

void test_f3(B *b) {
  int &ir = f3(b);
}

// When both conversions convert to 'id' or 'Class', pick the most
// specific type to convert from.
void f4(id);

void test_f4(ConvertsTo<B*> toB, ConvertsTo<C*> toC, ConvertsToBoth toBoth) {
  f4(toB);
  f4(toC);
  f4(toBoth);
}

void f5(id<P1>);

void test_f5(ConvertsTo<B*> toB, ConvertsTo<C*> toC, ConvertsToBoth toBoth) {
  f5(toB);
  f5(toC);
  f5(toBoth);
}


// A conversion to an a non-id object pointer type is better than a 
// conversion to qualified 'id'.
int &f6(A*);
float &f6(id<P1>);

void test_f6(B *b) {
  int &ir = f6(b);
}
