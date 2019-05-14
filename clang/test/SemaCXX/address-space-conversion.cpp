// RUN: %clang_cc1 -fsyntax-only -verify %s

// This test checks for the various conversions and casting operations
// with address-space-qualified pointers.

struct A { virtual ~A() {} };
struct B : A { };

typedef void *void_ptr;
typedef void __attribute__((address_space(1))) *void_ptr_1;
typedef void __attribute__((address_space(2))) *void_ptr_2;

typedef int *int_ptr;
typedef int __attribute__((address_space(1))) *int_ptr_1;
typedef int __attribute__((address_space(2))) *int_ptr_2;

typedef A *A_ptr;
typedef A __attribute__((address_space(1))) *A_ptr_1;
typedef A __attribute__((address_space(2))) *A_ptr_2;

typedef B *B_ptr;
typedef B __attribute__((address_space(1))) *B_ptr_1;
typedef B __attribute__((address_space(2))) *B_ptr_2;

void test_const_cast(int_ptr ip, int_ptr_1 ip1, int_ptr_2 ip2,
                     A_ptr ap, A_ptr_1 ap1, A_ptr_2 ap2,
                     const int *cip, 
                     const int __attribute__((address_space(1))) *cip1) {
  // Cannot use const_cast to cast between address spaces, add an
  // address space, or remove an address space.
  (void)const_cast<int_ptr>(ip1); // expected-error{{is not allowed}}
  (void)const_cast<int_ptr>(ip2); // expected-error{{is not allowed}}
  (void)const_cast<int_ptr_1>(ip); // expected-error{{is not allowed}}
  (void)const_cast<int_ptr_1>(ip2); // expected-error{{is not allowed}}
  (void)const_cast<int_ptr_2>(ip); // expected-error{{is not allowed}}
  (void)const_cast<int_ptr_2>(ip1); // expected-error{{is not allowed}}

  (void)const_cast<A_ptr>(ap1); // expected-error{{is not allowed}}
  (void)const_cast<A_ptr>(ap2); // expected-error{{is not allowed}}
  (void)const_cast<A_ptr_1>(ap); // expected-error{{is not allowed}}
  (void)const_cast<A_ptr_1>(ap2); // expected-error{{is not allowed}}
  (void)const_cast<A_ptr_2>(ap); // expected-error{{is not allowed}}
  (void)const_cast<A_ptr_2>(ap1); // expected-error{{is not allowed}}

  // It's acceptable to cast away constness.
  (void)const_cast<int_ptr>(cip);
  (void)const_cast<int_ptr_1>(cip1);
}

void test_static_cast(void_ptr vp, void_ptr_1 vp1, void_ptr_2 vp2,
                      A_ptr ap, A_ptr_1 ap1, A_ptr_2 ap2,
                      B_ptr bp, B_ptr_1 bp1, B_ptr_2 bp2) {
  // Well-formed upcast
  (void)static_cast<A_ptr>(bp);
  (void)static_cast<A_ptr_1>(bp1);
  (void)static_cast<A_ptr_2>(bp2);

  // Well-formed downcast
  (void)static_cast<B_ptr>(ap);
  (void)static_cast<B_ptr_1>(ap1);
  (void)static_cast<B_ptr_2>(ap2);

  // Well-formed cast to/from void
  (void)static_cast<void_ptr>(ap);
  (void)static_cast<void_ptr_1>(ap1);
  (void)static_cast<void_ptr_2>(ap2);
  (void)static_cast<A_ptr>(vp);
  (void)static_cast<A_ptr_1>(vp1);
  (void)static_cast<A_ptr_2>(vp2);
  
  // Ill-formed upcasts
  (void)static_cast<A_ptr>(bp1); // expected-error{{is not allowed}}
  (void)static_cast<A_ptr>(bp2); // expected-error{{is not allowed}}
  (void)static_cast<A_ptr_1>(bp); // expected-error{{is not allowed}}
  (void)static_cast<A_ptr_1>(bp2); // expected-error{{is not allowed}}
  (void)static_cast<A_ptr_2>(bp); // expected-error{{is not allowed}}
  (void)static_cast<A_ptr_2>(bp1); // expected-error{{is not allowed}}

  // Ill-formed downcasts
  (void)static_cast<B_ptr>(ap1); // expected-error{{casts away qualifiers}}
  (void)static_cast<B_ptr>(ap2); // expected-error{{casts away qualifiers}}
  (void)static_cast<B_ptr_1>(ap); // expected-error{{casts away qualifiers}}
  (void)static_cast<B_ptr_1>(ap2); // expected-error{{casts away qualifiers}}
  (void)static_cast<B_ptr_2>(ap); // expected-error{{casts away qualifiers}}
  (void)static_cast<B_ptr_2>(ap1); // expected-error{{casts away qualifiers}}

  // Ill-formed cast to/from void
  (void)static_cast<void_ptr>(ap1); // expected-error{{is not allowed}}
  (void)static_cast<void_ptr>(ap2); // expected-error{{is not allowed}}
  (void)static_cast<void_ptr_1>(ap); // expected-error{{is not allowed}}
  (void)static_cast<void_ptr_1>(ap2); // expected-error{{is not allowed}}
  (void)static_cast<void_ptr_2>(ap); // expected-error{{is not allowed}}
  (void)static_cast<void_ptr_2>(ap1); // expected-error{{is not allowed}}
  (void)static_cast<A_ptr>(vp1); // expected-error{{casts away qualifiers}}
  (void)static_cast<A_ptr>(vp2); // expected-error{{casts away qualifiers}}
  (void)static_cast<A_ptr_1>(vp); // expected-error{{casts away qualifiers}}
  (void)static_cast<A_ptr_1>(vp2); // expected-error{{casts away qualifiers}}
  (void)static_cast<A_ptr_2>(vp); // expected-error{{casts away qualifiers}}
  (void)static_cast<A_ptr_2>(vp1); // expected-error{{casts away qualifiers}}
}

void test_dynamic_cast(A_ptr ap, A_ptr_1 ap1, A_ptr_2 ap2,
                       B_ptr bp, B_ptr_1 bp1, B_ptr_2 bp2) {
  // Well-formed upcast
  (void)dynamic_cast<A_ptr>(bp);
  (void)dynamic_cast<A_ptr_1>(bp1);
  (void)dynamic_cast<A_ptr_2>(bp2);

  // Well-formed downcast
  (void)dynamic_cast<B_ptr>(ap);
  (void)dynamic_cast<B_ptr_1>(ap1);
  (void)dynamic_cast<B_ptr_2>(ap2);

  // Ill-formed upcasts
  (void)dynamic_cast<A_ptr>(bp1); // expected-error{{casts away qualifiers}}
  (void)dynamic_cast<A_ptr>(bp2); // expected-error{{casts away qualifiers}}
  (void)dynamic_cast<A_ptr_1>(bp); // expected-error{{casts away qualifiers}}
  (void)dynamic_cast<A_ptr_1>(bp2); // expected-error{{casts away qualifiers}}
  (void)dynamic_cast<A_ptr_2>(bp); // expected-error{{casts away qualifiers}}
  (void)dynamic_cast<A_ptr_2>(bp1); // expected-error{{casts away qualifiers}}

  // Ill-formed downcasts
  (void)dynamic_cast<B_ptr>(ap1); // expected-error{{casts away qualifiers}}
  (void)dynamic_cast<B_ptr>(ap2); // expected-error{{casts away qualifiers}}
  (void)dynamic_cast<B_ptr_1>(ap); // expected-error{{casts away qualifiers}}
  (void)dynamic_cast<B_ptr_1>(ap2); // expected-error{{casts away qualifiers}}
  (void)dynamic_cast<B_ptr_2>(ap); // expected-error{{casts away qualifiers}}
  (void)dynamic_cast<B_ptr_2>(ap1); // expected-error{{casts away qualifiers}}
}

void test_reinterpret_cast(void_ptr vp, void_ptr_1 vp1, void_ptr_2 vp2,
                           A_ptr ap, A_ptr_1 ap1, A_ptr_2 ap2,
                           B_ptr bp, B_ptr_1 bp1, B_ptr_2 bp2,
                           const void __attribute__((address_space(1))) * cvp1) {
  // reinterpret_cast can't be used to cast to a different address space unless they are matching (i.e. overlapping).
  (void)reinterpret_cast<A_ptr>(ap1); // expected-error{{reinterpret_cast from 'A_ptr_1' (aka '__attribute__((address_space(1))) A *') to 'A_ptr' (aka 'A *') is not allowed}}
  (void)reinterpret_cast<A_ptr>(ap2); // expected-error{{reinterpret_cast from 'A_ptr_2' (aka '__attribute__((address_space(2))) A *') to 'A_ptr' (aka 'A *') is not allowed}}
  (void)reinterpret_cast<A_ptr>(bp);
  (void)reinterpret_cast<A_ptr>(bp1); // expected-error{{reinterpret_cast from 'B_ptr_1' (aka '__attribute__((address_space(1))) B *') to 'A_ptr' (aka 'A *') is not allowed}}
  (void)reinterpret_cast<A_ptr>(bp2); // expected-error{{reinterpret_cast from 'B_ptr_2' (aka '__attribute__((address_space(2))) B *') to 'A_ptr' (aka 'A *') is not allowed}}
  (void)reinterpret_cast<A_ptr>(vp);
  (void)reinterpret_cast<A_ptr>(vp1);   // expected-error{{reinterpret_cast from 'void_ptr_1' (aka '__attribute__((address_space(1))) void *') to 'A_ptr' (aka 'A *') is not allowed}}
  (void)reinterpret_cast<A_ptr>(vp2);   // expected-error{{reinterpret_cast from 'void_ptr_2' (aka '__attribute__((address_space(2))) void *') to 'A_ptr' (aka 'A *') is not allowed}}
  (void)reinterpret_cast<A_ptr_1>(ap);  // expected-error{{reinterpret_cast from 'A_ptr' (aka 'A *') to 'A_ptr_1' (aka '__attribute__((address_space(1))) A *') is not allowed}}
  (void)reinterpret_cast<A_ptr_1>(ap2); // expected-error{{reinterpret_cast from 'A_ptr_2' (aka '__attribute__((address_space(2))) A *') to 'A_ptr_1' (aka '__attribute__((address_space(1))) A *') is not allowed}}
  (void)reinterpret_cast<A_ptr_1>(bp);  // expected-error{{reinterpret_cast from 'B_ptr' (aka 'B *') to 'A_ptr_1' (aka '__attribute__((address_space(1))) A *') is not allowed}}
  (void)reinterpret_cast<A_ptr_1>(bp1);
  (void)reinterpret_cast<A_ptr_1>(bp2); // expected-error{{reinterpret_cast from 'B_ptr_2' (aka '__attribute__((address_space(2))) B *') to 'A_ptr_1' (aka '__attribute__((address_space(1))) A *') is not allowed}}
  (void)reinterpret_cast<A_ptr_1>(vp);  // expected-error{{reinterpret_cast from 'void_ptr' (aka 'void *') to 'A_ptr_1' (aka '__attribute__((address_space(1))) A *') is not allowed}}
  (void)reinterpret_cast<A_ptr_1>(vp1);
  (void)reinterpret_cast<A_ptr_1>(vp2); // expected-error{{reinterpret_cast from 'void_ptr_2' (aka '__attribute__((address_space(2))) void *') to 'A_ptr_1' (aka '__attribute__((address_space(1))) A *') is not allowed}}

  // ... but don't try to cast away constness!
  (void)reinterpret_cast<A_ptr_2>(cvp1); // expected-error{{casts away qualifiers}}
}

void test_cstyle_cast(void_ptr vp, void_ptr_1 vp1, void_ptr_2 vp2,
                      A_ptr ap, A_ptr_1 ap1, A_ptr_2 ap2,
                      B_ptr bp, B_ptr_1 bp1, B_ptr_2 bp2,
                      const void __attribute__((address_space(1))) *cvp1) {
  // C-style casts are the wild west of casts.
  (void)(A_ptr)(ap1);
  (void)(A_ptr)(ap2);
  (void)(A_ptr)(bp);
  (void)(A_ptr)(bp1);
  (void)(A_ptr)(bp2);
  (void)(A_ptr)(vp);
  (void)(A_ptr)(vp1);
  (void)(A_ptr)(vp2);
  (void)(A_ptr_1)(ap);
  (void)(A_ptr_1)(ap2);
  (void)(A_ptr_1)(bp);
  (void)(A_ptr_1)(bp1);
  (void)(A_ptr_1)(bp2);
  (void)(A_ptr_1)(vp);
  (void)(A_ptr_1)(vp1);
  (void)(A_ptr_1)(vp2);
  (void)(A_ptr_2)(cvp1);
}

void test_implicit_conversion(void_ptr vp, void_ptr_1 vp1, void_ptr_2 vp2,
                              A_ptr ap, A_ptr_1 ap1, A_ptr_2 ap2,
                              B_ptr bp, B_ptr_1 bp1, B_ptr_2 bp2) {
  // Well-formed conversions
  void_ptr vpA = ap;
  void_ptr_1 vp_1A = ap1;
  void_ptr_2 vp_2A = ap2;
  A_ptr ap_A = bp;
  A_ptr_1 ap_A1 = bp1;
  A_ptr_2 ap_A2 = bp2;

  // Ill-formed conversions
  void_ptr vpB = ap1; // expected-error{{cannot initialize a variable of type}}
  void_ptr_1 vp_1B = ap2; // expected-error{{cannot initialize a variable of type}}
  A_ptr ap_B = bp1; // expected-error{{cannot initialize a variable of type}}
  A_ptr_1 ap_B1 = bp2; // expected-error{{cannot initialize a variable of type}}
}
