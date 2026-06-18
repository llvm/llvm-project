// Regression test for https://github.com/llvm/llvm-project/issues/198133
//
// A friend function-template specialization declared inside a class template
// is serialized into a PCH.  When the class template is later instantiated
// while loading the PCH, the friend specialization could be deserialized
// re-entrantly (VisitFriendDecl -> VisitFunctionDecl -> ... -> VisitFunctionDecl
// for the same specialization).  This used to trip the assertion
//   "already deserialized this template specialization"
// in ASTReaderDecl::VisitFunctionDecl for non-modules (PCH) builds.

// RUN: %clang_cc1 -std=c++17 -x c++-header -emit-pch %s -o %t.pch
// RUN: %clang_cc1 -std=c++17 -include-pch %t.pch %s -fsyntax-only -verify

#ifndef HEADER
#define HEADER

template <bool = false> int get_extents(const int &);
template <typename> struct BoundingBoxBase {
  BoundingBoxBase(int) {}
  friend int get_extents<>(const int &);
};
template <class> struct BoundingBox3Base {
  BoundingBox3Base();
};
struct BoundingBoxf : BoundingBoxBase<int> {
  BoundingBoxf(int points) : BoundingBoxBase(points) {}
};

#else

// expected-no-diagnostics
void f() { BoundingBox3Base<int> build_volume; }

#endif
