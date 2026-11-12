// RUN: %check_clang_tidy %s cppcoreguidelines-pro-type-cstyle-cast %t

void reinterpretcast() {
  int i = 0;
  void *j;
  j = (int*)j;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not use C-style cast to convert between unrelated types [cppcoreguidelines-pro-type-cstyle-cast]
}

void constcast() {
  int* i;
  const int* j;
  i = (int*)j;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not use C-style cast to cast away constness
  j = (const int*)i; // OK, const added
  (void)j; // OK, not a const_cast
}

void const_and_reinterpret() {
  int* i;
  const void* j;
  i = (int*)j;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not use C-style cast to convert between unrelated types
}

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

  auto P0 = (Derived*)new Base();
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use C-style cast to downcast from a base to a derived class

  const Base* B0;
  auto PC0 = (const Derived*)(B0);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: do not use C-style cast to downcast from a base to a derived class

  auto P1 = (Base*)new Derived(); // OK, upcast to a public base
  auto P2 = (Base*)new MultiDerived(); // OK, upcast to a public base
  auto P3 = (Base2*)new MultiDerived(); // OK, upcast to a public base
}

void pointers_polymorphic() {

  auto PP0 = (PolymorphicDerived*)new PolymorphicBase();
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: do not use C-style cast to downcast from a base to a derived class; use dynamic_cast instead
  // CHECK-FIXES: auto PP0 = dynamic_cast<PolymorphicDerived*>(new PolymorphicBase());

  const PolymorphicBase* B0;
  auto PPC0 = (const PolymorphicDerived*)B0;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: do not use C-style cast to downcast from a base to a derived class; use dynamic_cast instead
  // CHECK-FIXES: auto PPC0 = dynamic_cast<const PolymorphicDerived*>(B0);


  auto B1 = (PolymorphicBase*)new PolymorphicDerived(); // OK, upcast to a public base
  auto B2 = (PolymorphicBase*)new PolymorphicMultiDerived(); // OK, upcast to a public base
  auto B3 = (Base*)new PolymorphicMultiDerived(); // OK, upcast to a public base
}

void arrays() {
  Base ArrayOfBase[10];
  auto A0 = (Derived*)ArrayOfBase;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use C-style cast to downcast from a base to a derived class
}

void arrays_polymorphic() {
  PolymorphicBase ArrayOfPolymorphicBase[10];
  auto AP0 = (PolymorphicDerived*)ArrayOfPolymorphicBase;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: do not use C-style cast to downcast from a base to a derived class; use dynamic_cast instead
  // CHECK-FIXES: auto AP0 = dynamic_cast<PolymorphicDerived*>(ArrayOfPolymorphicBase);
}

void references() {
  Base B0;
  auto R0 = (Derived&)B0;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use C-style cast to downcast from a base to a derived class
  Base& RefToBase = B0;
  auto R1 = (Derived&)RefToBase;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use C-style cast to downcast from a base to a derived class

  const Base& ConstRefToBase = B0;
  auto RC1 = (const Derived&)ConstRefToBase;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: do not use C-style cast to downcast from a base to a derived class


  Derived RD1;
  auto R2 = (Base&)RD1; // OK, upcast to a public base
}

void references_polymorphic() {
  PolymorphicBase B0;
  auto RP0 = (PolymorphicDerived&)B0;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: do not use C-style cast to downcast from a base to a derived class; use dynamic_cast instead
  // CHECK-FIXES: auto RP0 = dynamic_cast<PolymorphicDerived&>(B0);

  PolymorphicBase& RefToPolymorphicBase = B0;
  auto RP1 = (PolymorphicDerived&)RefToPolymorphicBase;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: do not use C-style cast to downcast from a base to a derived class; use dynamic_cast instead
  // CHECK-FIXES: auto RP1 = dynamic_cast<PolymorphicDerived&>(RefToPolymorphicBase);

  const PolymorphicBase& ConstRefToPolymorphicBase = B0;
  auto RPC2 = (const PolymorphicDerived&)(ConstRefToPolymorphicBase);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: do not use C-style cast to downcast from a base to a derived class; use dynamic_cast instead
  // CHECK-FIXES: auto RPC2 = dynamic_cast<const PolymorphicDerived&>(ConstRefToPolymorphicBase);

  PolymorphicDerived d1;
  auto RP2 = (PolymorphicBase&)d1; // OK, upcast to a public base
}

template<class B, class D>
void templ() {
  auto B0 = (B*)new D();
}

void templ_bad_call() {
  templ<Derived, Base>(); //FIXME: this should trigger a warning
}

void templ_good_call() {
  templ<Base, Derived>(); // OK, upcast to a public base
}
