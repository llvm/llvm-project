// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodule-cache-path %t -I %S/Inputs %s -verify

// Importing modules which add declarations to a pre-existing non-imported
// overload set does not currently work.
// XFAIL: *

namespace N6 {
  char &f(char);
}

namespace N8 { }

@__experimental_modules_import namespaces_left;
@__experimental_modules_import namespaces_right;

void test() {
  int &ir1 = N1::f(1);
  int &ir2 = N2::f(1);
  int &ir3 = N3::f(1);
  float &fr1 = N1::f(1.0f);
  float &fr2 = N2::f(1.0f);
  double &dr1 = N2::f(1.0);
  double &dr2 = N3::f(1.0);
}

// Test namespaces merged without a common first declaration.
namespace N5 {
  char &f(char);
}

namespace N10 { 
  int &f(int);
}

void testMerged() {
  int &ir1 = N5::f(17);
  int &ir2 = N6::f(17);
  int &ir3 = N7::f(17);
  double &fr1 = N5::f(1.0);
  double &fr2 = N6::f(1.0);
  double &fr3 = N7::f(1.0);
  char &cr1 = N5::f('a');
  char &cr2 = N6::f('b');
}

// Test merging of declarations within namespaces that themselves were
// merged without a common first declaration.
void testMergedMerged() {
  int &ir1 = N8::f(17);
  int &ir2 = N9::f(17);
  int &ir3 = N10::f(17);
}

// Test merging when using anonymous namespaces, which does not
// actually perform any merging.
// other file: expected-note{{passing argument to parameter here}}
void testAnonymousNotMerged() {
  N11::consumeFoo(N11::getFoo()); // expected-error{{cannot initialize a parameter of type 'N11::<anonymous>::Foo *' with an rvalue of type 'N11::<anonymous>::Foo *'}}
  N12::consumeFoo(N12::getFoo()); // expected-error{{cannot initialize a parameter of type 'N12::<anonymous>::Foo *' with an rvalue of type 'N12::<anonymous>::Foo *'}}  
}


// other file: expected-note{{passing argument to parameter here}}
