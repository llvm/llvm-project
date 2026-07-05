// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct ConstCopy {
  ConstCopy();
  ConstCopy &operator=(const ConstCopy&);
};

struct NonConstCopy {
  NonConstCopy();
  NonConstCopy &operator=(NonConstCopy&);
};

struct VirtualInheritsNonConstCopy : virtual NonConstCopy { 
  VirtualInheritsNonConstCopy();
  VirtualInheritsNonConstCopy &operator=(const VirtualInheritsNonConstCopy&);
};

struct ImplicitNonConstCopy1 : NonConstCopy {  // expected-note{{the implicit copy assignment operator}}
  ImplicitNonConstCopy1();
};

struct ImplicitNonConstCopy2 { // expected-note{{the implicit copy assignment operator}}
  ImplicitNonConstCopy2();
  NonConstCopy ncc;
};

struct ImplicitNonConstCopy3 { // expected-note{{the implicit copy assignment operator}}
  ImplicitNonConstCopy3();
  NonConstCopy ncc_array[2][3];
};

struct ImplicitNonConstCopy4 : VirtualInheritsNonConstCopy { 
  ImplicitNonConstCopy4();
};

void test_non_const_copy(const ImplicitNonConstCopy1 &cincc1,
                         const ImplicitNonConstCopy2 &cincc2,
                         const ImplicitNonConstCopy3 &cincc3,
                         const ImplicitNonConstCopy4 &cincc4,
                         const VirtualInheritsNonConstCopy &vincc) {
  (void)sizeof(ImplicitNonConstCopy1() = cincc1); // expected-error{{no viable overloaded '='}}
  (void)sizeof(ImplicitNonConstCopy2() = cincc2); // expected-error{{no viable overloaded '='}}
  (void)sizeof(ImplicitNonConstCopy3() = cincc3); // expected-error{{no viable overloaded '='}}
  (void)sizeof(ImplicitNonConstCopy4() = cincc4); // okay
  (void)sizeof(VirtualInheritsNonConstCopy() = vincc);
}
