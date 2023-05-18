// RUN: %check_clang_tidy  -check-suffixes=NSTRICT,STRICT %s cppcoreguidelines-pro-type-static-cast-downcast %t
// RUN: %check_clang_tidy  -check-suffix=NSTRICT %s cppcoreguidelines-pro-type-static-cast-downcast %t -- -config="{CheckOptions: {StrictMode: false}}"

class Base {
};

class Derived : public Base {
};

class Base2 {
};

class MultiDerived : public Base, public Base2 {
};

class PolymorphicBase {
public:
  virtual ~PolymorphicBase();
};

class PolymorphicDerived : public PolymorphicBase {
};

class PolymorphicMultiDerived : public Base, public PolymorphicBase {
};

void pointers() {

  auto P0 = static_cast<Derived*>(new Base());
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:13: warning: do not use static_cast to downcast from a base to a derived class [cppcoreguidelines-pro-type-static-cast-downcast]

  const Base* B0;
  auto PC0 = static_cast<const Derived*>(B0);
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:14: warning: do not use static_cast to downcast from a base to a derived class [cppcoreguidelines-pro-type-static-cast-downcast]

  auto P1 = static_cast<Base*>(new Derived()); // OK, upcast to a public base
  auto P2 = static_cast<Base*>(new MultiDerived()); // OK, upcast to a public base
  auto P3 = static_cast<Base2*>(new MultiDerived()); // OK, upcast to a public base
}

void pointers_polymorphic() {

  auto PP0 = static_cast<PolymorphicDerived*>(new PolymorphicBase());
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:14: warning: do not use static_cast to downcast from a base to a derived class; use dynamic_cast instead [cppcoreguidelines-pro-type-static-cast-downcast]
  // CHECK-FIXES-NSTRICT: auto PP0 = dynamic_cast<PolymorphicDerived*>(new PolymorphicBase());

  const PolymorphicBase* B0;
  auto PPC0 = static_cast<const PolymorphicDerived*>(B0);
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:15: warning: do not use static_cast to downcast from a base to a derived class; use dynamic_cast instead [cppcoreguidelines-pro-type-static-cast-downcast]
  // CHECK-FIXES-NSTRICT: auto PPC0 = dynamic_cast<const PolymorphicDerived*>(B0);


  auto B1 = static_cast<PolymorphicBase*>(new PolymorphicDerived()); // OK, upcast to a public base
  auto B2 = static_cast<PolymorphicBase*>(new PolymorphicMultiDerived()); // OK, upcast to a public base
  auto B3 = static_cast<Base*>(new PolymorphicMultiDerived()); // OK, upcast to a public base
}

void arrays() {
  Base ArrayOfBase[10];
  auto A0 = static_cast<Derived*>(ArrayOfBase);
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:13: warning: do not use static_cast to downcast from a base to a derived class [cppcoreguidelines-pro-type-static-cast-downcast]
}

void arrays_polymorphic() {
  PolymorphicBase ArrayOfPolymorphicBase[10];
  auto AP0 = static_cast<PolymorphicDerived*>(ArrayOfPolymorphicBase);
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:14: warning: do not use static_cast to downcast from a base to a derived class; use dynamic_cast instead
  // CHECK-FIXES-NSTRICT: auto AP0 = dynamic_cast<PolymorphicDerived*>(ArrayOfPolymorphicBase);
}

void references() {
  Base B0;
  auto R0 = static_cast<Derived&>(B0);
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:13: warning: do not use static_cast to downcast from a base to a derived class [cppcoreguidelines-pro-type-static-cast-downcast]
  Base& RefToBase = B0;
  auto R1 = static_cast<Derived&>(RefToBase);
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:13: warning: do not use static_cast to downcast from a base to a derived class [cppcoreguidelines-pro-type-static-cast-downcast]

  const Base& ConstRefToBase = B0;
  auto RC1 = static_cast<const Derived&>(ConstRefToBase);
  // CHECK-MESSAGES-STRICT: :[[@LINE-1]]:14: warning: do not use static_cast to downcast from a base to a derived class [cppcoreguidelines-pro-type-static-cast-downcast]


  Derived RD1;
  auto R2 = static_cast<Base&>(RD1); // OK, upcast to a public base
}

void references_polymorphic() {
  PolymorphicBase B0;
  auto RP0 = static_cast<PolymorphicDerived&>(B0);
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:14: warning: do not use static_cast to downcast from a base to a derived class; use dynamic_cast instead
  // CHECK-FIXES-NSTRICT: auto RP0 = dynamic_cast<PolymorphicDerived&>(B0);

  PolymorphicBase& RefToPolymorphicBase = B0;
  auto RP1 = static_cast<PolymorphicDerived&>(RefToPolymorphicBase);
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:14: warning: do not use static_cast to downcast from a base to a derived class; use dynamic_cast instead [cppcoreguidelines-pro-type-static-cast-downcast]
  // CHECK-FIXES-NSTRICT: auto RP1 = dynamic_cast<PolymorphicDerived&>(RefToPolymorphicBase);

  const PolymorphicBase& ConstRefToPolymorphicBase = B0;
  auto RPC2 = static_cast<const PolymorphicDerived&>(ConstRefToPolymorphicBase);
  // CHECK-MESSAGES-NSTRICT: :[[@LINE-1]]:15: warning: do not use static_cast to downcast from a base to a derived class; use dynamic_cast instead [cppcoreguidelines-pro-type-static-cast-downcast]
  // CHECK-FIXES-NSTRICT: auto RPC2 = dynamic_cast<const PolymorphicDerived&>(ConstRefToPolymorphicBase);

  PolymorphicDerived d1;
  auto RP2 = static_cast<PolymorphicBase&>(d1); // OK, upcast to a public base
}

template<class B, class D>
void templ() {
  auto B0 = static_cast<B*>(new D());
}

void templ_bad_call() {
  templ<Derived, Base>(); //FIXME: this should trigger a warning
}

void templ_good_call() {
  templ<Base, Derived>(); // OK, upcast to a public base
}
