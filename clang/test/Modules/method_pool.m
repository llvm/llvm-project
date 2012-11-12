// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodule-cache-path %t -fmodules -I %S/Inputs %s -verify

@__experimental_modules_import MethodPoolA;


// in other file: // expected-note{{using}}




// in other file: expected-note{{also found}}

void testMethod1(id object) {
  [object method1]; 
}

void testMethod2(id object) {
  [object method2:1];
} 

@__experimental_modules_import MethodPoolB;

void testMethod1Again(id object) {
  [object method1];
}

void testMethod2Again(id object) {
  [object method2:1]; // expected-warning{{multiple methods named 'method2:' found}}
}
