// RUN: %clang_cc1 -fsyntax-only -verify %s

#define restrict __restrict__
typedef int* ptr;
void test1(ptr p, const ptr cp, restrict ptr rp, const restrict ptr crp,
           volatile ptr vp, const volatile ptr cvp, restrict volatile ptr rvp,
           const restrict volatile ptr crvp) {
  ptr& p1 = p;
  ptr& p2 = cp; // expected-error {{drops 'const' qualifier}}
  ptr& p3 = rp; // expected-error {{drops 'restrict' qualifier}}
  ptr& p4 = crp; // expected-error {{drops 'const' and 'restrict' qualifiers}}
  ptr& p5 = vp; // expected-error {{drops 'volatile' qualifier}}
  ptr& p6 = cvp; // expected-error {{drops 'const' and 'volatile' qualifiers}}
  ptr& p7 = rvp; // expected-error {{drops 'restrict' and 'volatile' qualifiers}}
  ptr& p8 = crvp; // expected-error {{drops 'const', 'restrict', and 'volatile' qualifiers}}

  const ptr& cp1 = p;
  const ptr& cp2 = cp;
  const ptr& cp3 = rp; // expected-error {{drops 'restrict' qualifier}}
  const ptr& cp4 = crp; // expected-error {{drops 'restrict' qualifier}}
  const ptr& cp5 = vp; // expected-error {{drops 'volatile' qualifier}}
  const ptr& cp6 = cvp; // expected-error {{drops 'volatile' qualifier}}
  const ptr& cp7 = rvp; // expected-error {{drops 'restrict' and 'volatile' qualifiers}}
  const ptr& cp8 = crvp; // expected-error {{drops 'restrict' and 'volatile' qualifiers}}

  const volatile ptr& cvp1 = p;
  const volatile ptr& cvp2 = cp;
  const volatile ptr& cvp3 = rp; // expected-error {{drops 'restrict' qualifier}}
  const volatile ptr& cvp4 = crp; // expected-error {{drops 'restrict' qualifier}}
  const volatile ptr& cvp5 = vp;
  const volatile ptr& cvp6 = cvp;
  const volatile ptr& cvp7 = rvp; // expected-error {{drops 'restrict' qualifier}}
  const volatile ptr& cvp8 = crvp; // expected-error {{drops 'restrict' qualifier}}

  const restrict volatile ptr& crvp1 = p;
  const restrict volatile ptr& crvp2 = cp;
  const restrict volatile ptr& crvp3 = rp;
  const restrict volatile ptr& crvp4 = crp;
  const restrict volatile ptr& crvp5 = vp;
  const restrict volatile ptr& crvp6 = cvp;
  const restrict volatile ptr& crvp7 = rvp;
  const restrict volatile ptr& crvp8 = crvp;
}
