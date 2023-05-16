// RUN: %clang_cc1 -fsyntax-only -Wmismatched-tags -verify %s
// RUN: not %clang_cc1 -fsyntax-only -Wmismatched-tags %s 2>&1 | FileCheck %s
class X; // expected-note 2{{here}}
typedef struct X * X_t; // expected-warning{{previously declared}}
union X { int x; float y; }; // expected-error{{use of 'X' with tag type that does not match previous declaration}}

template<typename T> struct Y; // expected-note{{did you mean class here?}}
template<class U> class Y { }; // expected-warning{{previously declared}}

template <typename>
struct Z {
  struct Z { // expected-error{{member 'Z' has the same name as its class}}
  };
};

class A;
class A;  // expected-note{{previous use is here}}
struct A;  // expected-warning{{struct 'A' was previously declared as a class}}

class B;  // expected-note{{did you mean struct here?}}
class B;  // expected-note{{previous use is here}}\
          // expected-note{{did you mean struct here?}}
struct B;  // expected-warning{{struct 'B' was previously declared as a class}}
struct B {};  // expected-warning{{'B' defined as a struct here but previously declared as a class}}

class C;  // expected-note{{previous use is here}}
struct C;  // expected-warning{{struct 'C' was previously declared as a class}}\
           // expected-note{{previous use is here}}\
           // expected-note{{did you mean class here?}}
class C;  // expected-warning{{class 'C' was previously declared as a struct}}\
          // expected-note{{previous use is here}}
struct C;  // expected-warning{{struct 'C' was previously declared as a class}}\
           // expected-note{{did you mean class here?}}
class C {};  // expected-warning{{'C' defined as a class here but previously declared as a struct}}

struct D {};  // expected-note{{previous definition is here}}\
              // expected-note{{previous use is here}}
class D {};  // expected-error{{redefinition of 'D'}}
struct D;
class D;  // expected-warning{{class 'D' was previously declared as a struct}}\
          // expected-note{{did you mean struct here?}}

class E;
class E;
class E {};
class E;

struct F;
struct F;
struct F {}; // expected-note {{previous use}}
struct F;
class F; // expected-warning {{previously declared as a struct}} expected-note {{did you mean struct}}

template<class U> class G;  // expected-note{{previous use is here}}\
                            // expected-note{{did you mean struct here?}}
template<class U> struct G;  // expected-warning{{struct template 'G' was previously declared as a class template}}
template<class U> struct G {};  // expected-warning{{'G' defined as a struct template here but previously declared as a class template}}

// Declarations from contexts where the warning is disabled are entirely
// ignored for the purpose of this warning.
struct J;
struct K; // expected-note {{previous use}}
struct L;
struct M; // expected-note {{previous use}}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmismatched-tags"
struct H;
class I {};
class J;
class K;
class L;
class M {};
#pragma clang diagnostic pop

class H; // expected-note {{previous use}}
struct H; // expected-warning {{previously declared as a class}}

struct I; // expected-note {{previous use}}
class I; // expected-warning {{previously declared as a struct}}

struct J;
class K; // expected-warning {{previously declared as a struct}}
struct L;
class M; // expected-warning {{previously declared as a struct}}

/*
*** 'X' messages ***
CHECK: warning: struct 'X' was previously declared as a class
CHECK: {{^}}typedef struct X * X_t;
CHECK: {{^}}        ^{{$}}
CHECK: note: previous use is here
CHECK: {{^}}class X;
CHECK: {{^}}      ^{{$}}
CHECK: error: use of 'X' with tag type that does not match previous declaration
CHECK: {{^}}union X { int x; float y; };
CHECK: {{^}}^~~~~{{$}}
CHECK: {{^}}class{{$}}
CHECK: note: previous use is here
CHECK: {{^}}class X;
CHECK: {{^}}      ^{{$}}
*** 'Y' messages ***
CHECK: warning: 'Y' defined as a class template here but
      previously declared as a struct template
CHECK: {{^}}template<class U> class Y { };
CHECK: {{^}}                  ^{{$}}
CHECK: note: did you mean class here?
CHECK: {{^}}template<typename T> struct Y;
CHECK: {{^}}                     ^~~~~~{{$}}
CHECK: {{^}}                     class{{$}}
*** 'A' messages ***
CHECK: warning: struct 'A' was previously declared as a class
CHECK: {{^}}struct A;
CHECK: {{^}}^{{$}}
CHECK: note: previous use is here
CHECK: {{^}}class A;
CHECK: {{^}}      ^{{$}}
*** 'B' messages ***
CHECK: warning: struct 'B' was previously declared as a class
CHECK: {{^}}struct B;
CHECK: {{^}}^{{$}}
CHECK: note: previous use is here
CHECK: {{^}}class B;
CHECK: {{^}}      ^{{$}}
CHECK: 'B' defined as a struct here but previously declared as a class
CHECK: {{^}}struct B {};
CHECK: {{^}}^{{$}}
CHECK: note: did you mean struct here?
CHECK: {{^}}class B;
CHECK: {{^}}^~~~~{{$}}
CHECK: {{^}}struct{{$}}
CHECK: note: did you mean struct here?
CHECK: {{^}}class B;
CHECK: {{^}}^~~~~{{$}}
CHECK: {{^}}struct{{$}}
*** 'C' messages ***
CHECK: warning: struct 'C' was previously declared as a class
CHECK: {{^}}struct C;
CHECK: {{^}}^{{$}}
CHECK: note: previous use is here
CHECK: {{^}}class C;
CHECK: {{^}}      ^{{$}}
CHECK: warning: class 'C' was previously declared as a struct
CHECK: {{^}}class C;
CHECK: {{^}}^{{$}}
CHECK: note: previous use is here
CHECK: {{^}}struct C;
CHECK: {{^}}       ^{{$}}
CHECK: warning: struct 'C' was previously declared as a class
CHECK: {{^}}struct C;
CHECK: {{^}}^{{$}}
CHECK: note: previous use is here
CHECK: {{^}}class C;
CHECK: {{^}}      ^{{$}}
CHECK: warning: 'C' defined as a class here but previously declared as a struct
CHECK: {{^}}class C {};
CHECK: {{^}}^{{$}}
CHECK: note: did you mean class here?
CHECK: {{^}}struct C;
CHECK: {{^}}^~~~~~{{$}}
CHECK: {{^}}class{{$}}
CHECK: note: did you mean class here?
CHECK: {{^}}struct C;
CHECK: {{^}}^~~~~~{{$}}
CHECK: {{^}}class{{$}}
*** 'D' messages ***
CHECK: error: redefinition of 'D'
CHECK: {{^}}class D {};
CHECK: {{^}}      ^{{$}}
CHECK: note: previous definition is here
CHECK: {{^}}struct D {};
CHECK: {{^}}       ^{{$}}
CHECK: warning: class 'D' was previously declared as a struct
CHECK: {{^}}class D;
CHECK: {{^}}^{{$}}
CHECK: note: previous use is here
CHECK: {{^}}struct D {};
CHECK: {{^}}       ^{{$}}
CHECK: note: did you mean struct here?
CHECK: {{^}}class D;
CHECK: {{^}}^~~~~{{$}}
CHECK: {{^}}struct{{$}}
*** 'E' messages ***
*** 'F' messages ***
*** 'G' messages ***
CHECK: warning: struct template 'G' was previously declared as a class template
CHECK: {{^}}template<class U> struct G;
CHECK: {{^}}                  ^{{$}}
CHECK: note: previous use is here
CHECK: {{^}}template<class U> class G;
CHECK: {{^}}                        ^{{$}}
CHECK: warning: 'G' defined as a struct template here but previously declared as a class template
CHECK: {{^}}template<class U> struct G {};
CHECK: {{^}}                  ^{{$}}
CHECK: note: did you mean struct here?
CHECK: {{^}}template<class U> class G;
CHECK: {{^}}                  ^~~~~
CHECK: {{^}}                  struct
*/
