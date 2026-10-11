// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-linux-gnu %s

// An array whose element type is incomplete when the array type is formed can
// only have its size checked once the element type is completed (#GH213855).

namespace GH213855 {
template <unsigned Size> struct S : public CBdVfsImpl { // expected-error {{expected class name}}
  double A[Size];
};
template <unsigned Size> struct SS {
  S<Size> A[Size]; // expected-error {{array is too large (147'573'944'137'180'895'432 bytes)}}
void foo() { SS<-123> ss; } // expected-error {{non-type template argument evaluates to -123, which cannot be narrowed to type 'unsigned int'}} \
                            // expected-note {{in instantiation of template class 'GH213855::SS<4294967173>' requested here}}
};
} // namespace GH213855

namespace array_variable {
template <unsigned Size> struct S { double A[Size]; };
S<4294967173u> arr[4294967173u]; // expected-error {{array is too large (147'573'944'137'180'895'432 bytes)}}
} // namespace array_variable

namespace dependent_context {
template <unsigned Size> struct S { double A[Size]; };

// Diagnosed once, at the template definition, not once per instantiation.
template <typename T> struct Parent {
  S<4294967173u> A[4294967173u]; // expected-error {{array is too large (147'573'944'137'180'895'432 bytes)}}
};
Parent<int> p1;
Parent<long> p2;

template <typename T> struct Parent2 {
  double A[2305843009213693952u]; // expected-error {{array is too large (2'305'843'009'213'693'952 elements)}}
};
Parent2<int> p3;
Parent2<long> p4;

template <typename T> struct ParentOK {
  double A[4294967173u]; // ~2^35 bytes, still representable.
};
ParentOK<int> ok;
} // namespace dependent_context

namespace incomplete_element_type {
struct Incomplete;
extern Incomplete ok[2];
extern Incomplete arr[4294967173];
extern Incomplete arr2[2][4294967173];
struct Incomplete { double A[4294967173]; };
Incomplete arr3[4294967173]; // expected-error {{array is too large (4'294'967'173 elements)}}

unsigned long n0 = sizeof(ok);
unsigned long n1 = sizeof(arr); // expected-error {{array is too large (147'573'944'137'180'895'432 bytes)}}
unsigned long n2 = sizeof(arr2); // expected-error {{array is too large (147'573'944'137'180'895'432 bytes)}}
} // namespace incomplete_element_type
