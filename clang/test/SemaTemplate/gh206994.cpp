// RUN: %clang_cc1 -std=c++23 -fsyntax-only -ferror-limit 19 -verify %s

// Reduced from a fuzzer-generated crash report. Every line is needed
// to trigger the specific error recovery state that causes the crash.

__detail __detail:); // expected-error {{unknown type name '__detail'}} \
                     // expected-error {{expected ';' after top level declarator}} \
                     // expected-error {{expected unqualified-id}}
template _Tptypename convertible_to_Tpcommon_reference_t_Tp_Up } // expected-error {{unknown type name '_Tptypename'; did you mean 'typename'?}} \
  // expected-error {{expected a qualified name after 'typename'}} \
  // expected-error {{variable cannot be defined in an explicit instantiation}} \
  // expected-error {{expected ';' after top level declarator}} \
  // expected-error {{extraneous closing brace}}
requires0template // expected-error {{unknown type name 'requires0template'}}
namespace ranges __iter_traits >; // expected-error {{expected unqualified-id}} \
  // expected-error {{expected '{'}} \
  // expected-error {{unknown type name '__iter_traits'}} \
  // expected-error {{expected unqualified-id}}
template  __iter_diff_tremove_cvref_t_Tp rangesiter_swap0 // expected-error {{unknown type name '__iter_diff_tremove_cvref_t_Tp'}} \
  // expected-error {{expected ';' after top level declarator}}
namespace detail { // expected-error {{variable cannot be defined in an explicit instantiation}}
template < typename > struct difference_type_ // expected-error {{expected ';' after struct}} \
  // expected-note {{'detail::difference_type_' declared here}}
}
template < typename T > using difference_type = difference_type_< T >; // expected-error {{no template named 'difference_type_'; did you mean 'detail::difference_type_'?}}
namespace detail {
template < typename T >
struct difference_type_ :  T // expected-error {{expected '{' after base class list}}
} difference_type alloc_limit = 4
// expected-error@* {{too many errors emitted, stopping now}}
