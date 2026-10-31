// RUN: %clang_cc1 -fsyntax-only -verify %s


namespace gh98199_0{
inline namespace A {
  namespace N {}
}
inline namespace B {
  namespace N {}
}
namespace N {   // expected-error {{reference to 'N' is ambiguous}}
    // expected-note@-6 {{candidate found by name lookup is 'gh98199_0::A::N'}}
    // expected-note@-4 {{candidate found by name lookup is 'gh98199_0::B::N'}}
  void f();
}
}

namespace gh98199_1{
inline namespace A{
    inline namespace A_ { namespace N {} }
    inline namespace A__ { namespace N {} }
}

inline namespace B{
    inline namespace B_ { namespace N {} }
    inline namespace B__ { namespace N {} }
}

namespace N{    // expected-error {{reference to 'N' is ambiguous}}
    // expected-note@-10 {{candidate found by name lookup is 'gh98199_1::A_::N'}}
    // expected-note@-10 {{candidate found by name lookup is 'gh98199_1::A__::N'}}
    // expected-note@-7 {{candidate found by name lookup is 'gh98199_1::B_::N'}}
    // expected-note@-7 {{candidate found by name lookup is 'gh98199_1::B__::N'}}
}
}

namespace gh98199_2{
namespace A{ namespace N{} }
using namespace A;
namespace N{}   // no error

inline namespace B{ namespace M{} }
namespace M{}   // no error
}

namespace gh98199_3{
namespace A{ namespace N{} }
namespace B{ namespace N{} }
using namespace A;
using namespace B;
namespace N{}   // no error
namespace A::N{};   // no error
}

namespace gh98199_4{
namespace A{
    inline namespace A_{
        inline namespace A__{
            namespace N{}
        }
    }
}
inline namespace B{
    inline namespace B_1{
        inline namespace B__{
            namespace N{}
        }
    }
    inline namespace B_2{
        namespace N{}
    }
    namespace N{    // expected-error {{reference to 'N' is ambiguous}}
        // expected-note@-7 {{candidate found by name lookup is 'gh98199_4::B_1::N'}}
        // expected-note@-4 {{candidate found by name lookup is 'gh98199_4::B_2::N'}}
    }
}

using namespace A;

namespace N{    // expected-error {{reference to 'N' is ambiguous}}
    // expected-note@-15 {{candidate found by name lookup is 'gh98199_4::B_1::N'}}
    // expected-note@-12 {{candidate found by name lookup is 'gh98199_4::B_2::N'}}
}
}

namespace gh98199_5{
inline namespace A { namespace N {} }
inline namespace B { namespace N {} }
inline namespace C { namespace N {} }
namespace N {}  // expected-error {{reference to 'N' is ambiguous}}
    // expected-note@-4 {{candidate found by name lookup is 'gh98199_5::A::N'}}
    // expected-note@-4 {{candidate found by name lookup is 'gh98199_5::B::N'}}
    // expected-note@-4 {{candidate found by name lookup is 'gh98199_5::C::N'}}
}

namespace gh98199_6{
namespace C{
  inline namespace A { namespace N {} }
  inline namespace B { namespace N {} }
}
namespace C{
  namespace N {}  // expected-error {{reference to 'N' is ambiguous}}
    // expected-note@-5 {{candidate found by name lookup is 'gh98199_6::C::A::N'}}
    // expected-note@-5 {{candidate found by name lookup is 'gh98199_6::C::B::N'}}
}

namespace C::N{    // expected-error {{reference to 'N' is ambiguous}}
    // expected-note@-10 {{candidate found by name lookup is 'gh98199_6::C::A::N'}}
    // expected-note@-10 {{candidate found by name lookup is 'gh98199_6::C::B::N'}}
}
}

namespace gh98199_7{
inline namespace A { namespace N {} }
namespace B { namespace N {} }
namespace N {}   // no error
}

namespace gh98199_8{
inline namespace A { inline namespace A_ { namespace N {} } }
inline namespace B { namespace N {} }
namespace N{  // expected-error {{reference to 'N' is ambiguous}}
    // expected-note@-3 {{candidate found by name lookup is 'gh98199_8::A::N'}}
    // expected-note@-3 {{candidate found by name lookup is 'gh98199_8::B::N'}}
}
}

namespace gh98199_9{
namespace { namespace N {} }
namespace N {}   // no error
}

namespace gh98199_10{
namespace {namespace N{} }
inline namespace A { namespace N{} }
namespace N{}   // no error
}

namespace gh98199_11{
namespace { namespace N{} }
namespace { inline namespace A{ namespace N{} } }
namespace N{}
namespace { namespace N{ // expected-error {{reference to 'N' is ambiguous}}
    // expected-note@-4 {{candidate found by name lookup is 'gh98199_11::(anonymous namespace)::N'}}
    // expected-note@-4 {{candidate found by name lookup is 'gh98199_11::(anonymous namespace)::A::N'}}
}   // anon namespace
}   // namespace N
}

namespace gh98199_12{
inline namespace A { namespace X {} namespace Y {} }
inline namespace B { namespace X {} namespace Y {} }
namespace X {}  // expected-error {{reference to 'X' is ambiguous}}
    // expected-note@-3 {{candidate found by name lookup is 'gh98199_12::A::X'}}
    // expected-note@-3 {{candidate found by name lookup is 'gh98199_12::B::X'}}
namespace Y {}  // expected-error {{reference to 'Y' is ambiguous}}
    // expected-note@-6 {{candidate found by name lookup is 'gh98199_12::A::Y'}}
    // expected-note@-6 {{candidate found by name lookup is 'gh98199_12::B::Y'}}
}

namespace gh98199_13{
inline namespace A { namespace N { int x; } }
namespace N{
    int x;  // expected-error {{redefinition of 'x'}}
    // expected-note@-3 {{previous definition is here}}
}
}

namespace gh98199_14{
namespace C {
  inline namespace A { namespace N {} }
  inline namespace B { namespace N {} }
}
namespace Other {
  using namespace C;
  namespace N {}   // no error
}
}
