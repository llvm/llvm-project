// RUN: %clang_cc1 -fsyntax-only -verify %s

struct X0 {
  void f0();
  void f1() const;
  void f2() volatile;
  void f3() const volatile;
};

void test_object_cvquals(void (X0::*pm)(),
                         void (X0::*pmc)() const,
                         void (X0::*pmv)() volatile,
                         void (X0::*pmcv)() const volatile,
                         X0 *p,
                         const X0 *pc,
                         volatile X0 *pv,
                         const volatile X0 *pcv,
                         X0 &o,
                         const X0 &oc,
                         volatile X0 &ov,
                         const volatile X0 &ocv) {
  (p->*pm)();
  (p->*pmc)();
  (p->*pmv)();
  (p->*pmcv)();

  (pc->*pm)(); // expected-error{{call to pointer to member function of type 'void ()' drops 'const' qualifier}}
  (pc->*pmc)();
  (pc->*pmv)(); // expected-error{{call to pointer to member function of type 'void () volatile' drops 'const' qualifier}}
  (pc->*pmcv)();

  (pv->*pm)(); // expected-error{{call to pointer to member function of type 'void ()' drops 'volatile' qualifier}}
  (pv->*pmc)(); // expected-error{{call to pointer to member function of type 'void () const' drops 'volatile' qualifier}}
  (pv->*pmv)();
  (pv->*pmcv)();

  (pcv->*pm)(); // expected-error{{call to pointer to member function of type 'void ()' drops 'const volatile' qualifiers}}
  (pcv->*pmc)(); // expected-error{{call to pointer to member function of type 'void () const' drops 'volatile' qualifier}}
  (pcv->*pmv)(); // expected-error{{call to pointer to member function of type 'void () volatile' drops 'const' qualifier}}
  (pcv->*pmcv)();

  (o.*pm)();
  (o.*pmc)();
  (o.*pmv)();
  (o.*pmcv)();

  (oc.*pm)(); // expected-error{{call to pointer to member function of type 'void ()' drops 'const' qualifier}}
  (oc.*pmc)();
  (oc.*pmv)(); // expected-error{{call to pointer to member function of type 'void () volatile' drops 'const' qualifier}}
  (oc.*pmcv)();

  (ov.*pm)(); // expected-error{{call to pointer to member function of type 'void ()' drops 'volatile' qualifier}}
  (ov.*pmc)(); // expected-error{{call to pointer to member function of type 'void () const' drops 'volatile' qualifier}}
  (ov.*pmv)();
  (ov.*pmcv)();

  (ocv.*pm)(); // expected-error{{call to pointer to member function of type 'void ()' drops 'const volatile' qualifiers}}
  (ocv.*pmc)(); // expected-error{{call to pointer to member function of type 'void () const' drops 'volatile' qualifier}}
  (ocv.*pmv)(); // expected-error{{call to pointer to member function of type 'void () volatile' drops 'const' qualifier}}
  (ocv.*pmcv)();
}
