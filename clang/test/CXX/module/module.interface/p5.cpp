// RUN: %clang_cc1 -std=c++2a %s -verify -pedantic-errors

export module p5;

int a;
static int sa; // expected-note {{target}}
void b();
static void sb(); // expected-note {{target}}
struct c {};
enum d {};
using e = int;
using f = c;
static union { int sg1, sg2; }; // expected-note {{target}}
namespace NS {}

template<typename> int ta;
template<typename> static int sta; // expected-note {{target}}
template<typename> void tb();
template<typename> static void stb(); // expected-note {{target}}
template<typename> struct tc {};
template<typename> using te = int;
template<typename> using tf = c;

namespace UnnamedNS {
  namespace {
    int a; // expected-note {{target}}
    static int sa; // expected-note {{target}}
    void b(); // expected-note {{target}}
    static void sb(); // expected-note {{target}}
    struct c {}; // expected-note {{target}}
    enum d {}; // expected-note {{target}}
    using e = int;
    using f = c;
    static union { int sg1, sg2; }; // expected-note {{target}}
    namespace NS {}

    template<typename> int ta; // expected-note {{target}}
    template<typename> static int sta; // expected-note {{target}}
    template<typename> void tb(); // expected-note {{target}}
    template<typename> static void stb(); // expected-note {{target}}
    template<typename> struct tc {}; // expected-note {{target}}
    template<typename> using te = int; // expected-note {{target}}
    template<typename> using tf = c; // expected-note {{target}}
  }
}

export { // expected-note 19{{here}}
  using ::a;
  using ::sa; // expected-error {{using declaration referring to 'sa' with internal linkage}}
  using ::b;
  using ::sb; // expected-error {{using declaration referring to 'sb' with internal linkage}}
  using ::c;
  using ::d;
  using ::e;
  using ::f;
  using ::sg1; // expected-error {{using declaration referring to 'sg1' with internal linkage}}

  using ::ta;
  using ::sta; // expected-error {{using declaration referring to 'sta' with internal linkage}}
  using ::tb;
  using ::stb; // expected-error {{using declaration referring to 'stb' with internal linkage}}
  using ::tc;
  using ::te;
  using ::tf;
  namespace NS2 = ::NS;

  namespace UnnamedNS {
    using UnnamedNS::a; // expected-error {{internal linkage}}
    using UnnamedNS::sa; // expected-error {{internal linkage}}
    using UnnamedNS::b; // expected-error {{internal linkage}}
    using UnnamedNS::sb; // expected-error {{internal linkage}}
    using UnnamedNS::c; // expected-error {{internal linkage}}
    using UnnamedNS::d; // expected-error {{internal linkage}}
    using UnnamedNS::e; // ok
    using UnnamedNS::f; // ok? using-declaration refers to alias-declaration,
                        // which does not have linkage (even though that then
                        // refers to a type that has internal linkage)
    using UnnamedNS::sg1; // expected-error {{internal linkage}}

    using UnnamedNS::ta; // expected-error {{internal linkage}}
    using UnnamedNS::sta; // expected-error {{internal linkage}}
    using UnnamedNS::tb; // expected-error {{internal linkage}}
    using UnnamedNS::stb; // expected-error {{internal linkage}}
    using UnnamedNS::tc; // expected-error {{internal linkage}}
    using UnnamedNS::te; // expected-error {{internal linkage}}
    using UnnamedNS::tf; // expected-error {{internal linkage}}
    namespace NS2 = UnnamedNS::NS; // ok (wording bug?)
  }
}
