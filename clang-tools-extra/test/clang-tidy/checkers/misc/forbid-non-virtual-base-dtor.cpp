// RUN: %check_clang_tidy %s misc-forbid-non-virtual-base-dtor %t

// should warn -> non-virtual base + derived has data
class A {};
class B : public A {
  // CHECK-MESSAGES: :[[@LINE-1]]:{{.*}}: warning: class 'B' inherits from 'A' which has a non-virtual destructor [misc-forbid-non-virtual-base-dtor]
  int x;
};

// shouldn't warn -> derived has no data
class C : public A {};

// shouldn't warn -> base has virtual destructor
class D {
public:
  virtual ~D() {}
};
class E : public D {
  int y;
};

// shouldn't crash -> incomplete then defined base
class F;
class F {};
class G : public F {
  // CHECK-MESSAGES: :[[@LINE-1]]:{{.*}}: warning: class 'G' inherits from 'F' which has a non-virtual destructor [misc-forbid-non-virtual-base-dtor]
  int z;
};

// shouldn't warn -> base has protected non-virtual destructor
class H {
protected:
  ~H() {}
};
class I : public H {
  int w;
};

// shouldn't warn -> private inheritance
class J : private A {
  int w;
};

// shouldn't warn -> protected inheritance
class K : protected A {
  int w;
};

// should warn for both bases
class M1 {};
class M2 {};
class M3 : public M1, public M2 {
  // CHECK-MESSAGES: :[[@LINE-1]]:{{.*}}: warning: class 'M3' inherits from 'M2' which has a non-virtual destructor [misc-forbid-non-virtual-base-dtor]
  int x;
};

// should warn only for non-virtual base
class M4 : public D, public A {
  // CHECK-MESSAGES: :[[@LINE-1]]:{{.*}}: warning: class 'M4' inherits from 'A' which has a non-virtual destructor [misc-forbid-non-virtual-base-dtor]
  int x;
};

// should warn -> non-virtual base instantiated with data member
template <typename T>
class TBase {};

template <typename T>
class TDerived : public TBase<T> {
  // CHECK-MESSAGES: :[[@LINE-1]]:{{.*}}: warning: class 'TDerived' inherits from 'TBase' which has a non-virtual destructor [misc-forbid-non-virtual-base-dtor]
  T x;
};
TDerived<int> td;

// shouldn't warn -> templated base with virtual destructor
template <typename T>
class TVBase {
public:
  virtual ~TVBase() {}
};

template <typename T>
class TVDerived : public TVBase<T> {
  T x;
};
TVDerived<int> tvd;

#define DERIVE_WITH_DATA(Derived, Base) \
  class Derived : public Base {         \
    int x;                              \
  };

DERIVE_WITH_DATA(MacroDerived, A)
// CHECK-MESSAGES: :[[@LINE-1]]:{{.*}}: warning: class 'MacroDerived' inherits from 'A' which has a non-virtual destructor [misc-forbid-non-virtual-base-dtor]

#define DERIVE_NO_DATA(Derived, Base) \
  class Derived : public Base {};

// shouldn't warn -> no data member
DERIVE_NO_DATA(MacroDerivedEmpty, A)
