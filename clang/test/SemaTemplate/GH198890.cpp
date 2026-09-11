// RUN: %clang_cc1 -fsyntax-only -std=c++14 -verify %s

// GitHub issue #198890: crash in VisitVarTemplatePartialSpecializationDecl
// when the primary member variable template fails to instantiate.
//
// When Outer<int> is instantiated, substituting typename T::type for T=int
// fails for both the primary template member and the partial specialization.
// The fix recovers from substitution failure in VisitVarDecl (marking the
// declaration invalid) so the VarTemplateDecl is still registered in the
// owner, allowing VisitVarTemplatePartialSpecializationDecl to find it via
// lookup without crashing.

namespace GH198890 {

template <typename T>
struct Outer {
  template <typename U>
  static typename T::type member; // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}

  template <typename U>
  static typename T::type member<U *>; // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}
};

template struct Outer<int>; // expected-note {{in instantiation of template class 'GH198890::Outer<int>' requested here}}

} // namespace GH198890
