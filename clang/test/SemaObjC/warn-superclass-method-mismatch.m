// RUN: %clang_cc1  -fsyntax-only -Wsuper-class-method-mismatch -verify %s

@interface Root
-(void) method_r: (char)ch : (float*)f1 : (int*) x; // expected-note {{previous declaration is here}}
@end

@class Sub;

@interface Base : Root
-(void) method: (int*) x; // expected-note {{previous declaration is here}}
-(void) method1: (Base*) x; // expected-note {{previous declaration is here}}
-(void) method2: (Sub*) x; // expected-note{{passing argument to parameter 'x' here}}
+ method3: (int)x1 : (Base *)x2 : (float)x3; // expected-note {{previous declaration is here}}
+ mathod4: (id)x1;
- method5: (int) x : (double) d; // expected-note {{previous declaration is here}}
- method6: (int) x : (float) d; // expected-note {{previous declaration is here}}
@end

struct A {
  int x,y,z;
};

@interface Sub : Base
-(void) method: (struct A*) a;	// expected-warning {{method parameter type 'struct A *' does not match super class method parameter type 'int *'}}
-(void) method1: (Sub*) x;	// expected-warning {{method parameter type 'Sub *' does not match super class method parameter type 'Base *'}}
-(void) method2: (Base*) x;	// no need to warn. At call point we warn if need be.
+ method3: (int)x1 : (Sub *)x2 : (float)x3;	// expected-warning {{method parameter type 'Sub *' does not match super class method parameter type 'Base *'}}
+ mathod4: (Base*)x1;
-(void) method_r: (char)ch : (float*)f1 : (Sub*) x; // expected-warning {{method parameter type 'Sub *' does not match super class method parameter type 'int *'}}
- method5: (int) x : (float) d; // expected-warning {{method parameter type 'float' does not match super class method parameter type 'double'}}
- method6: (int) x : (double) d; // expected-warning {{method parameter type 'double' does not match super class method parameter type 'float'}}
@end

void f(Base *base, Sub *sub) {
  int x;
  [base method:&x];  // warn. if base is actually 'Sub' it will use -[Sub method] with wrong arguments
  
  Base *b;
  [base method1:b]; // if base is actuall 'Sub'  it will use [Sub method1] with wrong argument.

  [base method2:b];  // expected-warning {{}}

  Sub *s;
  [base method2:s]; // if base is actually 'Sub' OK. Either way OK.
  
}




