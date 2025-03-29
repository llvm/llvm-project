// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -fixit -x c++ %t
// RUN: %clang_cc1 -fsyntax-only -pedantic -Werror -x c++ %t
// RUN: grep test_string %t

namespace std {
  template<typename T> class basic_string { // expected-note 3{{'basic_string' declared here}}
  public:
    int find(const char *substr); // expected-note{{'find' declared here}}
    static const int npos = -1; // expected-note{{'npos' declared here}}
  };

  typedef basic_string<char> string; // expected-note 2{{'string' declared here}}
}

namespace otherstd { // expected-note 2{{'otherstd' declared here}} \
                     // expected-note{{namespace 'otherstd' defined here}}
  using namespace std;
}

using namespace std;

other_std::strng str1; // expected-error{{use of undeclared identifier 'other_std'; did you mean 'otherstd'?}} \
// expected-error{{no type named 'strng' in namespace 'otherstd'; did you mean 'string'?}}
tring str2; // expected-error{{unknown type name 'tring'; did you mean 'string'?}}

::other_std::string str3; // expected-error{{no member named 'other_std' in the global namespace; did you mean 'otherstd'?}}

float area(float radius, // expected-note{{'radius' declared here}}
           float pi) {
  return radious * pi; // expected-error{{did you mean 'radius'?}}
}

using namespace othestd; // expected-error{{no namespace named 'othestd'; did you mean 'otherstd'?}}
namespace blargh = otherstd; // expected-note 3{{namespace 'blargh' defined here}}
using namespace ::blarg; // expected-error{{no namespace named 'blarg' in the global namespace; did you mean 'blargh'?}}

namespace wibble = blarg; // expected-error{{no namespace named 'blarg'; did you mean 'blargh'?}}
namespace wobble = ::blarg; // expected-error{{no namespace named 'blarg' in the global namespace; did you mean 'blargh'?}}

bool test_string(std::string s) {
  basc_string<char> b1; // expected-error{{no template named 'basc_string'; did you mean 'basic_string'?}}
  std::basic_sting<char> b2; // expected-error{{no template named 'basic_sting' in namespace 'std'; did you mean 'basic_string'?}}
  (void)b1;
  (void)b2;
  return s.fnd("hello") // expected-error{{no member named 'fnd' in 'std::basic_string<char>'; did you mean 'find'?}}
    == std::string::pos; // expected-error{{no member named 'pos' in 'std::basic_string<char>'; did you mean 'npos'?}}
}

struct Base { };
struct Derived : public Base { // expected-note{{base class 'Base' specified here}}
  int member; // expected-note 3{{'member' declared here}}

  Derived() : base(), // expected-error{{initializer 'base' does not name a non-static data member or base class; did you mean the base class 'Base'?}}
              ember() { } // expected-error{{initializer 'ember' does not name a non-static data member or base class; did you mean the member 'member'?}}

  int getMember() const {
    return ember; // expected-error{{use of undeclared identifier 'ember'; did you mean 'member'?}}
  }

  int &getMember();
};

int &Derived::getMember() {
  return ember; // expected-error{{use of undeclared identifier 'ember'; did you mean 'member'?}}
}

typedef int Integer; // expected-note{{'Integer' declared here}}
int global_value; // expected-note{{'global_value' declared here}}

int foo() {
  integer * i = 0; // expected-error{{unknown type name 'integer'; did you mean 'Integer'?}}
  unsinged *ptr = 0; // expected-error{{use of undeclared identifier 'unsinged'; did you mean 'unsigned'?}}
  return *i + *ptr + global_val; // expected-error{{use of undeclared identifier 'global_val'; did you mean 'global_value'?}}
}

namespace nonstd {
  typedef std::basic_string<char> yarn; // expected-note 2 {{'nonstd::yarn' declared here}}
  int narf; // expected-note{{'nonstd::narf' declared here}}
}

yarn str4; // expected-error{{unknown type name 'yarn'; did you mean 'nonstd::yarn'?}}
wibble::yarn str5; // expected-error{{no type named 'yarn' in namespace 'otherstd'; did you mean 'nonstd::yarn'?}}

namespace another {
  template<typename T> class wide_string {}; // expected-note {{'another::wide_string' declared here}}
}
int poit() {
  nonstd::basic_string<char> str; // expected-error{{no template named 'basic_string' in namespace 'nonstd'; did you mean simply 'basic_string'?}}
  nonstd::wide_string<char> str2; // expected-error{{no template named 'wide_string' in namespace 'nonstd'; did you mean 'another::wide_string'?}}
  return wibble::narf; // expected-error{{no member named 'narf' in namespace 'otherstd'; did you mean 'nonstd::narf'?}}
}

namespace check_bool {
  void f() {
    Bool b; // expected-error{{use of undeclared identifier 'Bool'; did you mean 'bool'?}}
  }
}

namespace outr {
}
namespace outer {
  namespace inner { // expected-note{{'outer::inner' declared here}} \
                    // expected-note{{namespace 'outer::inner' defined here}} \
                    // expected-note{{'inner' declared here}}
    int i;
  }
}

using namespace outr::inner; // expected-error{{no namespace named 'inner' in namespace 'outr'; did you mean 'outer::inner'?}}

void func() {
  outr::inner::i = 3; // expected-error{{no member named 'inner' in namespace 'outr'; did you mean 'outer::inner'?}}
  outer::innr::i = 4; // expected-error{{no member named 'innr' in namespace 'outer'; did you mean 'inner'?}}
}

struct base {
};
struct derived : base {
  int i;
};

void func2() {
  derived d;
  // FIXME: we should offer a fix here. We do if the 'i' is misspelled, but we don't do name qualification changes
  //        to replace base::i with derived::i as we would for other qualified name misspellings.
  // d.base::i = 3;
}

class A {
  void bar(int);
};
void bar(int, int);  // expected-note{{'::bar' declared here}}
void A::bar(int x) {
  bar(x, 5);  // expected-error{{too many arguments to function call, expected 1, have 2; did you mean '::bar'?}}
}
