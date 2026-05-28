// RUN: %clang_cc1 -fsyntax-only -std=c++14 -verify %s

// GitHub issue #198890
// https://github.com/llvm/llvm-project/issues/198890
//
// VisitVarTemplatePartialSpecializationDecl used hard assert() calls where the
// analogous VisitClassTemplatePartialSpecializationDecl used graceful null
// checks and return nullptr. In a release build (no assertions), the assert was
// compiled out and a null-pointer dereference (SIGSEGV) followed.
//
// The crash is triggered when the primary member variable template fails to
// instantiate (returning nullptr from VisitVarTemplateDecl), so that the
// subsequent lookup for the instantiated VarTemplateDecl in Owner comes back
// empty. Without the fix, VisitVarTemplatePartialSpecializationDecl would then
// dereference a null / invalid pointer via Found.front().

namespace GH198890 {

// T::type does not exist for plain types like int or void.
// When Outer<int> is instantiated, substituting typename T::type for T=int
// fails, VisitVarTemplateDecl returns nullptr, and the instantiated
// VarTemplateDecl is never added to Owner.  The partial specialisation is
// visited next; Owner->lookup("member") returns empty, which previously
// triggered the assert / SIGSEGV.

template <typename T>
struct Outer {
  template <typename U>
  static typename T::type member; // expected-error {{type 'int' cannot be used prior to '::' because it has no members}}

  template <typename U>
  static typename T::type member<U *>; // partial specialization
};

template struct Outer<int>; // expected-note {{in instantiation of template class 'GH198890::Outer<int>' requested here}}

} // namespace GH198890
