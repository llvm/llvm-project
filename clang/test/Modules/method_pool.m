// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs %s -verify


@import MethodPoolA;

@interface D
- (void)method5:(D*)obj;
@end

// expected-note@Inputs/MethodPoolA.h:7{{using}}
// expected-note@Inputs/MethodPoolB.h:12{{also found}}

void testMethod1(id object) {
  [object method1]; 
}

void testMethod2(id object) {
  [object method2:1];
} 

void testMethod4(id object) {
  [object method4]; // expected-warning{{instance method '-method4' not found (return type defaults to 'id')}}
} 

void testMethod5(id object, D* d) {
  [object method5:d];
}

@import MethodPoolB;

void testMethod1Again(id object) {
  [object method1];
}

void testMethod2Again(id object) {
  [object method2:1]; // expected-warning{{multiple methods named 'method2:' found}}
}

void testMethod3(id object) {
  [object method3]; // expected-warning{{instance method '-method3' not found (return type defaults to 'id')}}
}

@import MethodPoolB.Sub;

void testMethod3Again(id object) {
  char *str = [object method3]; // okay: only found in MethodPoolB.Sub
}

void testMethod6(id object) {
  [object method6];
}

@import MethodPoolA.Sub;

void testMethod3AgainAgain(id object) {
  [object method3]; // expected-warning{{multiple methods named 'method3' found}}
  // expected-note@Inputs/MethodPoolBSub.h:2{{using}}
  // expected-note@Inputs/MethodPoolASub.h:2{{also found}}
}

void testMethod4Again(id object) {
  [object method4];
} 

void testMethod5Again(id object, D* d) {
  [object method5:d];
}
