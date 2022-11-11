// RUN: %check_clang_tidy %s cppcoreguidelines-slicing %t

class Base {
  int i;
  void f() {}
  virtual void g() {}
};

class DerivedWithMemberVariables : public Base {
  void f();
  int j;
};

class TwiceDerivedWithNoMemberVariables : public DerivedWithMemberVariables {
  void f();
};

class DerivedWithOverride : public Base {
  void f();
  void g() override {}
};

class TwiceDerivedWithNoOverride : public DerivedWithOverride {
  void f();
};

void TakesBaseByValue(Base base);

DerivedWithMemberVariables ReturnsDerived();

void positivesWithMemberVariables() {
  DerivedWithMemberVariables b;
  Base a{b};
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: slicing object from type 'DerivedWithMemberVariables' to 'Base' discards {{[0-9]*}} bytes of state [cppcoreguidelines-slicing]
  a = b;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: slicing object from type 'DerivedWithMemberVariables' to 'Base' discards {{[0-9]*}} bytes of state
  TakesBaseByValue(b);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: slicing object from type 'DerivedWithMemberVariables' to 'Base' discards {{[0-9]*}} bytes of state

  TwiceDerivedWithNoMemberVariables c;
  a = c;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: slicing object from type 'TwiceDerivedWithNoMemberVariables' to 'Base' discards {{[0-9]*}} bytes of state

  a = ReturnsDerived();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: slicing object from type 'DerivedWithMemberVariables' to 'Base' discards {{[0-9]*}} bytes of state
}

void positivesWithOverride() {
  DerivedWithOverride b;
  Base a{b};
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: slicing object from type 'DerivedWithOverride' to 'Base' discards override 'g'
  a = b;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: slicing object from type 'DerivedWithOverride' to 'Base' discards override 'g'
  TakesBaseByValue(b);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: slicing object from type 'DerivedWithOverride' to 'Base' discards override 'g'

  TwiceDerivedWithNoOverride c;
  a = c;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: slicing object from type 'DerivedWithOverride' to 'Base' discards override 'g'
}

void TakesBaseByReference(Base &base);

class DerivedThatAddsVirtualH : public Base {
  virtual void h();
};

class DerivedThatOverridesH : public DerivedThatAddsVirtualH {
  void h() override;
};

void negatives() {
  // OK, simple copying from the same type.
  Base a;
  TakesBaseByValue(a);
  DerivedWithMemberVariables b;
  DerivedWithMemberVariables c{b};
  b = c;

  // OK, derived type does not have extra state.
  TwiceDerivedWithNoMemberVariables d;
  DerivedWithMemberVariables e{d};
  e = d;

  // OK, derived does not override any method.
  TwiceDerivedWithNoOverride f;
  DerivedWithOverride g{f};
  g = f;

  // OK, no copying.
  TakesBaseByReference(d);
  TakesBaseByReference(f);

  // Derived type overrides methods, but these methods are not in the base type,
  // so cannot be called accidentally. Right now this triggers, but we might
  // want to allow it.
  DerivedThatOverridesH h;
  a = h;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: slicing object from type 'DerivedThatOverridesH' to 'Base' discards override 'h'
}
