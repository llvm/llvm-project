// RUN: %clang_cc1 -triple x86_64-apple-macos10.10 %s                 -verify=c,expected
// RUN: %clang_cc1 -triple x86_64-apple-macos10.10 %s -xobjective-c++ -verify=cxx,expected
// RUN: %clang_cc1 -triple x86_64-apple-macos10.10 %s -fobjc-arc      -verify=c,arc,expected

typedef signed char BOOL;
#define YES __objc_yes
#define NO __objc_no

@interface NSNumber
+(instancetype)numberWithChar:(char)value;
+(instancetype)numberWithInt:(int)value;
+(instancetype)numberWithDouble:(double)value;
+(instancetype)numberWithBool:(BOOL)value;
@end

void test(void) {
  NSNumber *n = YES; // expected-error{{numeric literal must be prefixed by '@'}}
  NSNumber *n1 = 1; // expected-error{{numeric literal must be prefixed by '@'}}

  NSNumber *n2 = NO; // c-warning{{expression which evaluates to zero treated as a null pointer constant}}
                     // cxx-error@-1{{numeric literal must be prefixed by '@'}}
  NSNumber *n3 = 0;
  NSNumber *n4 = 0.0; // expected-error{{numeric literal must be prefixed by '@'}}

  NSNumber *n5 = '\0'; // c-warning{{expression which evaluates to zero treated as a null pointer constant}}
                       // cxx-error@-1{{numeric literal must be prefixed by '@'}}


  NSNumber *n6 = '1'; // expected-error{{numeric literal must be prefixed by '@'}}

  int i;
  NSNumber *n7 = i; // c-warning{{incompatible integer to pointer conversion}}
                    // arc-error@-1{{implicit conversion of 'int' to 'NSNumber *' is disallowed with ARC}}
                    // cxx-error@-2{{cannot initialize a variable of type 'NSNumber *' with an lvalue of type 'int'}}

  id n8 = 1; // c-warning{{incompatible integer to pointer conversion}}
             // arc-error@-1{{implicit conversion of 'int' to 'id' is disallowed with ARC}}
             // cxx-error@-2{{cannot initialize a variable of type 'id' with an rvalue of type 'int'}}
}
