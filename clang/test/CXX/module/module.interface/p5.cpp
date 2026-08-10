// RUN: %clang_cc1 -std=c++2a %s -verify -pedantic-errors

export module p5;

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

export { // expected-note 28{{here}}
  using ::a; // expected-error {{using declaration referring to 'a' with module linkage cannot be exported}}
  using ::sa; // expected-error {{using declaration referring to 'sa' with internal linkage}}
  using ::b; // expected-error {{using declaration referring to 'b' with module linkage cannot be exported}}
  using ::sb; // expected-error {{using declaration referring to 'sb' with internal linkage}}
  using ::c; // expected-error {{using declaration referring to 'c' with module linkage cannot be exported}}
  using ::d; // expected-error {{using declaration referring to 'd' with module linkage cannot be exported}}
  using ::e;
  using ::f;
  using ::sg1; // expected-error {{using declaration referring to 'sg1' with internal linkage}}

  using ::ta; // expected-error {{using declaration referring to 'ta' with module linkage cannot be exported}}
  using ::sta; // expected-error {{using declaration referring to 'sta' with internal linkage}}
  using ::tb; // expected-error {{using declaration referring to 'tb' with module linkage cannot be exported}}
  using ::stb; // expected-error {{using declaration referring to 'stb' with internal linkage}}
  using ::tc; // expected-error {{using declaration referring to 'tc' with module linkage cannot be exported}}
  using ::te; // expected-error {{using declaration referring to 'te' with module linkage cannot be exported}}
  using ::tf; // expected-error {{using declaration referring to 'tf' with module linkage cannot be exported}}
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
