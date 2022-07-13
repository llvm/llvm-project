// RUN: %clang_cc1 -std=c++11 -fsyntax-only -triple %itanium_abi_triple -verify -Wreinterpret-base-class -Wno-unused-volatile-lvalue %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -triple %ms_abi_triple -DMSABI -verify -Wreinterpret-base-class -Wno-unused-volatile-lvalue %s

// RUN: not %clang_cc1 -std=c++11 -fsyntax-only -triple %itanium_abi_triple -fdiagnostics-parseable-fixits -Wreinterpret-base-class -Wno-unused-volatile-lvalue %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -std=c++11 -fsyntax-only -triple %ms_abi_triple -fdiagnostics-parseable-fixits -Wreinterpret-base-class -Wno-unused-volatile-lvalue %s 2>&1 | FileCheck %s

// PR 13824
class A {
};
class DA : public A {
};
class DDA : public DA {
};
class DAo : protected A {
};
class DAi : private A {
};

class DVA : public virtual A {
};
class DDVA : public virtual DA {
};
class DMA : public virtual A, public virtual DA { //expected-warning{{direct base 'A' is inaccessible due to ambiguity:\n    class DMA -> class A\n    class DMA -> class DA -> class A}}
};

class B;

struct C {
  // Do not fail on incompletely-defined classes.
  decltype(reinterpret_cast<C *>(0)) foo;
  decltype(reinterpret_cast<A *>((C *) 0)) bar;
  decltype(reinterpret_cast<C *>((A *) 0)) baz;
};

void reinterpret_not_defined_class(B *b, C *c) {
  // Should not fail if class has no definition.
  (void)*reinterpret_cast<C *>(b);
  (void)*reinterpret_cast<B *>(c);

  (void)reinterpret_cast<C &>(*b);
  (void)reinterpret_cast<B &>(*c);
}

// Do not fail on erroneous classes with fields of incompletely-defined types.
// Base class is malformed.
namespace BaseMalformed {
  struct A; // expected-note {{forward declaration of 'BaseMalformed::A'}}
  struct B {
    A a; // expected-error {{field has incomplete type 'BaseMalformed::A'}}
  };
  struct C : public B {} c;
  B *b = reinterpret_cast<B *>(&c);
} // end anonymous namespace

// Child class is malformed.
namespace ChildMalformed {
  struct A; // expected-note {{forward declaration of 'ChildMalformed::A'}}
  struct B {};
  struct C : public B {
    A a; // expected-error {{field has incomplete type 'ChildMalformed::A'}}
  } c;
  B *b = reinterpret_cast<B *>(&c);
} // end anonymous namespace

// Base class outside upcast base-chain is malformed.
namespace BaseBaseMalformed {
  struct A; // expected-note {{forward declaration of 'BaseBaseMalformed::A'}}
  struct Y {};
  struct X { A a; }; // expected-error {{field has incomplete type 'BaseBaseMalformed::A'}}
  struct B : Y, X {};
  struct C : B {} c;
  B *p = reinterpret_cast<B*>(&c);
}

namespace InheritanceMalformed {
  struct A; // expected-note {{forward declaration of 'InheritanceMalformed::A'}}
  struct B : A {}; // expected-error {{base class has incomplete type}}
  struct C : B {} c;
  B *p = reinterpret_cast<B*>(&c);
}

// Virtual base class outside upcast base-chain is malformed.
namespace VBaseMalformed{
  struct A; // expected-note {{forward declaration of 'VBaseMalformed::A'}}
  struct X { A a; };  // expected-error {{field has incomplete type 'VBaseMalformed::A'}}
  struct B : public virtual X {};
  struct C : B {} c;
  B *p = reinterpret_cast<B*>(&c);
}

void reinterpret_not_updowncast(A *pa, const A *pca, A &a, const A &ca) {
  (void)*reinterpret_cast<C *>(pa);
  (void)*reinterpret_cast<const C *>(pa);
  (void)*reinterpret_cast<volatile C *>(pa);
  (void)*reinterpret_cast<const volatile C *>(pa);

  (void)*reinterpret_cast<const C *>(pca);
  (void)*reinterpret_cast<const volatile C *>(pca);

  (void)reinterpret_cast<C &>(a);
  (void)reinterpret_cast<const C &>(a);
  (void)reinterpret_cast<volatile C &>(a);
  (void)reinterpret_cast<const volatile C &>(a);

  (void)reinterpret_cast<const C &>(ca);
  (void)reinterpret_cast<const volatile C &>(ca);
}

void reinterpret_pointer_downcast(A *a, const A *ca) {
  (void)*reinterpret_cast<DA *>(a);
  (void)*reinterpret_cast<const DA *>(a);
  (void)*reinterpret_cast<volatile DA *>(a);
  (void)*reinterpret_cast<const volatile DA *>(a);

  (void)*reinterpret_cast<const DA *>(ca);
  (void)*reinterpret_cast<const volatile DA *>(ca);

  (void)*reinterpret_cast<DDA *>(a);
  (void)*reinterpret_cast<DAo *>(a);
  (void)*reinterpret_cast<DAi *>(a);
  // expected-warning@+2 {{'reinterpret_cast' to class 'DVA *' from its virtual base 'A *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while downcasting}}
  (void)*reinterpret_cast<DVA *>(a);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:26}:"static_cast"

  // expected-warning@+2 {{'reinterpret_cast' to class 'DDVA *' from its virtual base 'A *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while downcasting}}
  (void)*reinterpret_cast<DDVA *>(a);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:26}:"static_cast"

  // expected-warning@+2 {{'reinterpret_cast' to class 'DMA *' from its virtual base 'A *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while downcasting}}
  (void)*reinterpret_cast<DMA *>(a);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:26}:"static_cast"
}

void reinterpret_reference_downcast(A a, A &ra, const A &cra) {
  (void)reinterpret_cast<DA &>(a);
  (void)reinterpret_cast<const DA &>(a);
  (void)reinterpret_cast<volatile DA &>(a);
  (void)reinterpret_cast<const volatile DA &>(a);

  (void)reinterpret_cast<DA &>(ra);
  (void)reinterpret_cast<const DA &>(ra);
  (void)reinterpret_cast<volatile DA &>(ra);
  (void)reinterpret_cast<const volatile DA &>(ra);

  (void)reinterpret_cast<const DA &>(cra);
  (void)reinterpret_cast<const volatile DA &>(cra);

  (void)reinterpret_cast<DDA &>(a);
  (void)reinterpret_cast<DAo &>(a);
  (void)reinterpret_cast<DAi &>(a);
  // expected-warning@+2 {{'reinterpret_cast' to class 'DVA &' from its virtual base 'A' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while downcasting}}
  (void)reinterpret_cast<DVA &>(a);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"

  // expected-warning@+2 {{'reinterpret_cast' to class 'DDVA &' from its virtual base 'A' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while downcasting}}
  (void)reinterpret_cast<DDVA &>(a);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"

  // expected-warning@+2 {{'reinterpret_cast' to class 'DMA &' from its virtual base 'A' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while downcasting}}
  (void)reinterpret_cast<DMA &>(a);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"
}

void reinterpret_pointer_upcast(DA *da, const DA *cda, DDA *dda, DAo *dao,
                                DAi *dai, DVA *dva, DDVA *ddva, DMA *dma) {
  (void)*reinterpret_cast<A *>(da);
  (void)*reinterpret_cast<const A *>(da);
  (void)*reinterpret_cast<volatile A *>(da);
  (void)*reinterpret_cast<const volatile A *>(da);

  (void)*reinterpret_cast<const A *>(cda);
  (void)*reinterpret_cast<const volatile A *>(cda);

  (void)*reinterpret_cast<A *>(dda);
  (void)*reinterpret_cast<DA *>(dda);
  (void)*reinterpret_cast<A *>(dao);
  (void)*reinterpret_cast<A *>(dai);
  // expected-warning@+2 {{'reinterpret_cast' from class 'DVA *' to its virtual base 'A *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while upcasting}}
  (void)*reinterpret_cast<A *>(dva);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:26}:"static_cast"

  // expected-warning@+2 {{'reinterpret_cast' from class 'DDVA *' to its virtual base 'A *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while upcasting}}
  (void)*reinterpret_cast<A *>(ddva);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:26}:"static_cast"

  // expected-warning@+2 {{'reinterpret_cast' from class 'DDVA *' to its virtual base 'DA *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while upcasting}}
  (void)*reinterpret_cast<DA *>(ddva);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:26}:"static_cast"

  // expected-warning@+2 {{'reinterpret_cast' from class 'DMA *' to its virtual base 'A *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while upcasting}}
  (void)*reinterpret_cast<A *>(dma);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:26}:"static_cast"

  // expected-warning@+2 {{'reinterpret_cast' from class 'DMA *' to its virtual base 'DA *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while upcasting}}
  (void)*reinterpret_cast<DA *>(dma);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:26}:"static_cast"
}

void reinterpret_reference_upcast(DA &da, const DA &cda, DDA &dda, DAo &dao,
                                  DAi &dai, DVA &dva, DDVA &ddva, DMA &dma) {
  (void)reinterpret_cast<A &>(da);
  (void)reinterpret_cast<const A &>(da);
  (void)reinterpret_cast<volatile A &>(da);
  (void)reinterpret_cast<const volatile A &>(da);

  (void)reinterpret_cast<const A &>(cda);
  (void)reinterpret_cast<const volatile A &>(cda);

  (void)reinterpret_cast<A &>(dda);
  (void)reinterpret_cast<DA &>(dda);
  (void)reinterpret_cast<A &>(dao);
  (void)reinterpret_cast<A &>(dai);
  // expected-warning@+2 {{'reinterpret_cast' from class 'DVA' to its virtual base 'A &' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while upcasting}}
  (void)reinterpret_cast<A &>(dva);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"

  // expected-warning@+2 {{'reinterpret_cast' from class 'DDVA' to its virtual base 'A &' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while upcasting}}
  (void)reinterpret_cast<A &>(ddva);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"

  // expected-warning@+2 {{'reinterpret_cast' from class 'DDVA' to its virtual base 'DA &' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while upcasting}}
  (void)reinterpret_cast<DA &>(ddva);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"

  // expected-warning@+2 {{'reinterpret_cast' from class 'DMA' to its virtual base 'A &' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while upcasting}}
  (void)reinterpret_cast<A &>(dma);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"

  // expected-warning@+2 {{'reinterpret_cast' from class 'DMA' to its virtual base 'DA &' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while upcasting}}
  (void)reinterpret_cast<DA &>(dma);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"
}

struct E {
  int x;
};

class F : public E {
  virtual int foo() { return x; }
};

class G : public F {
};

class H : public E, public A {
};

class I : virtual public F {
};

typedef const F * K;
typedef volatile K L;

void different_subobject_downcast(E *e, F *f, A *a) {
  // expected-warning@+2 {{'reinterpret_cast' to class 'F *' from its base at non-zero offset 'E *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while downcasting}}
  (void)reinterpret_cast<F *>(e);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"

  // expected-warning@+2 {{'reinterpret_cast' to class 'G *' from its base at non-zero offset 'E *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while downcasting}}
  (void)reinterpret_cast<G *>(e);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"

  (void)reinterpret_cast<H *>(e);
  // expected-warning@+2 {{'reinterpret_cast' to class 'I *' from its virtual base 'E *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while downcasting}}
  (void)reinterpret_cast<I *>(e);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"


  (void)reinterpret_cast<G *>(f);
  // expected-warning@+2 {{'reinterpret_cast' to class 'I *' from its virtual base 'F *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while downcasting}}
  (void)reinterpret_cast<I *>(f);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"

#ifdef MSABI
  // In MS ABI mode, A is at non-zero offset in H.
  // expected-warning@+3 {{'reinterpret_cast' to class 'H *' from its base at non-zero offset 'A *' behaves differently from 'static_cast'}}
  // expected-note@+2 {{use 'static_cast'}}
#endif
  (void)reinterpret_cast<H *>(a);

  // expected-warning@+2 {{'reinterpret_cast' to class 'K' (aka 'const F *') from its base at non-zero offset 'E *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while downcasting}}
  (void)reinterpret_cast<L>(e);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"
}

void different_subobject_upcast(F *f, G *g, H *h, I *i) {
  // expected-warning@+2 {{'reinterpret_cast' from class 'F *' to its base at non-zero offset 'E *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while upcasting}}
  (void)reinterpret_cast<E *>(f);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"

  (void)reinterpret_cast<F *>(g);
  // expected-warning@+2 {{'reinterpret_cast' from class 'G *' to its base at non-zero offset 'E *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while upcasting}}
  (void)reinterpret_cast<E *>(g);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"

  (void)reinterpret_cast<E *>(h);

#ifdef MSABI
  // In MS ABI mode, A is at non-zero offset in H.
  // expected-warning@+3 {{'reinterpret_cast' from class 'H *' to its base at non-zero offset 'A *' behaves differently from 'static_cast'}}
  // expected-note@+2 {{use 'static_cast'}}
#endif
  (void)reinterpret_cast<A *>(h);

  // expected-warning@+2 {{'reinterpret_cast' from class 'I *' to its virtual base 'F *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while upcasting}}
  (void)reinterpret_cast<F *>(i);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"

  // expected-warning@+2 {{'reinterpret_cast' from class 'I *' to its virtual base 'E *' behaves differently from 'static_cast'}}
  // expected-note@+1 {{use 'static_cast' to adjust the pointer correctly while upcasting}}
  (void)reinterpret_cast<E *>(i);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:25}:"static_cast"
}
