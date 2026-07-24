// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c++17 -Rnullsafe-evidence %s -verify

struct Base {
  virtual ~Base();
};
struct Derived : Base {
  int value;
};

void takesNonnull(Derived *_Nonnull);

Derived *returnDynamic(Base *_Nonnull p) {
  return dynamic_cast<Derived *>(p); // expected-remark{{returns nullable}}
}

Derived *returnStatic(Derived *_Nonnull p) { // expected-remark{{function 'returnStatic' always returns a non-null pointer}}
  return static_cast<Derived *>(p); // expected-remark{{returns nonnull}}
}

void dynamicCastPropagation(Base *_Nonnull p) {
  Derived *d = dynamic_cast<Derived *>(p);
  d->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  takesNonnull(dynamic_cast<Derived *>(p)); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check before the call}}
}

void preservingCastControls(Derived *_Nonnull p) {
  Derived *d = static_cast<Derived *>(p);
  d->value = 1;
  takesNonnull(static_cast<Derived *>(p));
}

void narrowedDynamicCastIsSafe(Base *_Nonnull p) {
  if (Derived *d = dynamic_cast<Derived *>(p))
    d->value = 1;
}

void callerStillWarns(Base *_Nonnull p) {
  returnDynamic(p)->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}} expected-remark-re{{parameter 'p' of 'returnDynamic' (declared at {{.*}}) called with nonnull argument}}
}
