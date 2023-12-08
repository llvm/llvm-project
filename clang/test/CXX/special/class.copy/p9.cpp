// RUN: %clang_cc1 -fsyntax-only -verify %s

struct ConstCopy {
  ConstCopy();
  ConstCopy(const ConstCopy&);
};

struct NonConstCopy {
  NonConstCopy();
  NonConstCopy(NonConstCopy&);
};

struct VirtualInheritsNonConstCopy : virtual NonConstCopy { 
  VirtualInheritsNonConstCopy();
  VirtualInheritsNonConstCopy(const VirtualInheritsNonConstCopy&);
};

struct ImplicitNonConstCopy1 : NonConstCopy { // expected-note {{candidate constructor}}
  ImplicitNonConstCopy1(); // expected-note {{candidate constructor}}
};

struct ImplicitNonConstCopy2 { // expected-note {{candidate constructor}}
  ImplicitNonConstCopy2(); // expected-note {{candidate constructor}}
  NonConstCopy ncc;
};

struct ImplicitNonConstCopy3 { // expected-note {{candidate constructor}}
  ImplicitNonConstCopy3(); // expected-note {{candidate constructor}}
  NonConstCopy ncc_array[2][3];
};

struct ImplicitNonConstCopy4 : VirtualInheritsNonConstCopy { // expected-note {{candidate constructor}}
  ImplicitNonConstCopy4(); // expected-note {{candidate constructor}}
};

void test_non_const_copy(const ImplicitNonConstCopy1 &cincc1,
                         const ImplicitNonConstCopy2 &cincc2,
                         const ImplicitNonConstCopy3 &cincc3,
                         const ImplicitNonConstCopy4 &cincc4) {
  (void)sizeof(ImplicitNonConstCopy1(cincc1)); // expected-error{{no matching conversion for functional-style cast from 'const ImplicitNonConstCopy1' to 'ImplicitNonConstCopy1'}}
  (void)sizeof(ImplicitNonConstCopy2(cincc2)); // expected-error{{no matching conversion for functional-style cast from 'const ImplicitNonConstCopy2' to 'ImplicitNonConstCopy2'}}
  (void)sizeof(ImplicitNonConstCopy3(cincc3)); // expected-error{{no matching conversion for functional-style cast from 'const ImplicitNonConstCopy3' to 'ImplicitNonConstCopy3'}}
  (void)sizeof(ImplicitNonConstCopy4(cincc4)); // expected-error{{no matching conversion for functional-style cast from 'const ImplicitNonConstCopy4' to 'ImplicitNonConstCopy4'}}
}
