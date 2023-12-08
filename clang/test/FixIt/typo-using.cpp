// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -fsyntax-only -fixit -x c++ %t
// RUN: %clang_cc1 -fsyntax-only -pedantic -Werror -x c++ %t
// RUN: grep using_suggestion_tyname_ty_dropped_specifier %t

// These tests have been separated from typo.cpp to keep the maximum typo
// correction counter from ticking over; this causes spurious failures.

namespace using_suggestion_ty {
namespace N { class AAA {}; } // expected-note {{'AAA' declared here}}
using N::AAB; // expected-error {{no member named 'AAB' in namespace 'using_suggestion_ty::N'; did you mean 'AAA'?}}
}

namespace using_suggestion_tyname_ty {
namespace N { class AAA {}; } // expected-note {{'AAA' declared here}}
using typename N::AAB; // expected-error {{no member named 'AAB' in namespace 'using_suggestion_tyname_ty::N'; did you mean 'AAA'?}}
}

namespace using_suggestion_val {
namespace N { void FFF() {} } // expected-note {{'FFF' declared here}}
using N::FFG; // expected-error {{no member named 'FFG' in namespace 'using_suggestion_val::N'; did you mean 'FFF'?}}
}

namespace using_suggestion_ty_dropped_specifier {
class ABC {}; // expected-note {{'::using_suggestion_ty_dropped_specifier::ABC' declared here}}
namespace N { }
using N::ABC; // expected-error {{no member named 'ABC' in namespace 'using_suggestion_ty_dropped_specifier::N'; did you mean '::using_suggestion_ty_dropped_specifier::ABC'?}}
}

namespace using_suggestion_tyname_ty_dropped_specifier {
class BCD {}; // expected-note {{'::using_suggestion_tyname_ty_dropped_specifier::BCD' declared here}}
namespace N { }
using typename N::BCD; // expected-error {{no member named 'BCD' in namespace 'using_suggestion_tyname_ty_dropped_specifier::N'; did you mean '::using_suggestion_tyname_ty_dropped_specifier::BCD'?}}
}

namespace using_suggestion_val_dropped_specifier {
void EFG() {} // expected-note {{'::using_suggestion_val_dropped_specifier::EFG' declared here}}
namespace N { }
using N::EFG; // expected-error {{no member named 'EFG' in namespace 'using_suggestion_val_dropped_specifier::N'; did you mean '::using_suggestion_val_dropped_specifier::EFG'?}}
}

namespace using_suggestion_member_ty {
class CCC { public: typedef int AAA; }; // expected-note {{'AAA' declared here}}
class DDD : public CCC { public: using CCC::AAB; }; // expected-error {{no member named 'AAB' in 'using_suggestion_member_ty::CCC'; did you mean 'AAA'?}}
}

namespace using_suggestion_member_val {
class CCC { public: void AAA() { } }; // expected-note {{'AAA' declared here}}
class DDD : public CCC { public: using CCC::AAB; }; // expected-error {{no member named 'AAB' in 'using_suggestion_member_val::CCC'; did you mean 'AAA'?}}
}

namespace using_suggestion_member_tyname_ty {
class CCC { public: typedef int AAA; }; // expected-note {{'AAA' declared here}}
class DDD : public CCC { public: using typename CCC::AAB; }; // expected-error {{no member named 'AAB' in 'using_suggestion_member_tyname_ty::CCC'; did you mean 'AAA'?}}
}
