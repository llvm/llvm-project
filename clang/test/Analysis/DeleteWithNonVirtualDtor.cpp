// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.cplusplus.DeleteWithNonVirtualDtor -std=c++11 -verify -analyzer-output=text %s

struct Virtual {
  virtual ~Virtual() {}
};

struct VDerived : public Virtual {};

struct NonVirtual {
  ~NonVirtual() {}
};

struct NVDerived : public NonVirtual {};
struct NVDoubleDerived : public NVDerived {};

struct Base {
  virtual void destroy() = 0;
};

class PrivateDtor final : public Base {
public:
  void destroy() { delete this; }
private:
  ~PrivateDtor() {}
};

struct ImplicitNV {
  virtual void f();
};

struct ImplicitNVDerived : public ImplicitNV {};

NVDerived *get();

NonVirtual *create() {
  NonVirtual *x = new NVDerived(); // expected-note{{Casting from 'NVDerived' to 'NonVirtual' here}}
  return x;
}

void sink(NonVirtual *x) {
  delete x; // expected-warning{{Destruction of a polymorphic object with no virtual destructor}}
  // expected-note@-1{{Destruction of a polymorphic object with no virtual destructor}}
}

void sinkCast(NonVirtual *y) {
  delete reinterpret_cast<NVDerived*>(y);
}

void sinkParamCast(NVDerived *z) {
  delete z;
}

void singleDerived() {
  NonVirtual *sd;
  sd = new NVDerived(); // expected-note{{Casting from 'NVDerived' to 'NonVirtual' here}}
  delete sd; // expected-warning{{Destruction of a polymorphic object with no virtual destructor}}
  // expected-note@-1{{Destruction of a polymorphic object with no virtual destructor}}
}

void singleDerivedArr() {
  NonVirtual *sda = new NVDerived[5]; // expected-note{{Casting from 'NVDerived' to 'NonVirtual' here}}
  delete[] sda; // expected-warning{{Destruction of a polymorphic object with no virtual destructor}}
  // expected-note@-1{{Destruction of a polymorphic object with no virtual destructor}}
}

void doubleDerived() {
  NonVirtual *dd = new NVDoubleDerived(); // expected-note{{Casting from 'NVDoubleDerived' to 'NonVirtual' here}}
  delete (dd); // expected-warning{{Destruction of a polymorphic object with no virtual destructor}}
  // expected-note@-1{{Destruction of a polymorphic object with no virtual destructor}}
}

void assignThroughFunction() {
  NonVirtual *atf = get(); // expected-note{{Casting from 'NVDerived' to 'NonVirtual' here}}
  delete atf; // expected-warning{{Destruction of a polymorphic object with no virtual destructor}}
  // expected-note@-1{{Destruction of a polymorphic object with no virtual destructor}}
}

void assignThroughFunction2() {
  NonVirtual *atf2;
  atf2 = get(); // expected-note{{Casting from 'NVDerived' to 'NonVirtual' here}}
  delete atf2; // expected-warning{{Destruction of a polymorphic object with no virtual destructor}}
  // expected-note@-1{{Destruction of a polymorphic object with no virtual destructor}}
}

void createThroughFunction() {
  NonVirtual *ctf = create(); // expected-note{{Calling 'create'}}
  // expected-note@-1{{Returning from 'create'}}
  delete ctf; // expected-warning {{Destruction of a polymorphic object with no virtual destructor}}
  // expected-note@-1{{Destruction of a polymorphic object with no virtual destructor}}
}

void deleteThroughFunction() {
  NonVirtual *dtf = new NVDerived(); // expected-note{{Casting from 'NVDerived' to 'NonVirtual' here}}
  sink(dtf); // expected-note{{Calling 'sink'}}
}

void singleCastCStyle() {
  NVDerived *sccs = new NVDerived();
  NonVirtual *sccs2 = (NonVirtual*)sccs; // expected-note{{Casting from 'NVDerived' to 'NonVirtual' here}}
  delete sccs2; // expected-warning{{Destruction of a polymorphic object with no virtual destructor}}
  // expected-note@-1{{Destruction of a polymorphic object with no virtual destructor}}
}

void doubleCastCStyle() {
  NonVirtual *dccs = new NVDerived(); // expected-note{{Casting from 'NVDerived' to 'NonVirtual' here}}
  NVDerived *dccs2 = (NVDerived*)dccs; // expected-note{{Casting from 'NonVirtual' to 'NVDerived' here}}
  dccs = (NonVirtual*)dccs2; // expected-note{{Casting from 'NVDerived' to 'NonVirtual' here}}
  delete dccs; // expected-warning{{Destruction of a polymorphic object with no virtual destructor}}
  // expected-note@-1{{Destruction of a polymorphic object with no virtual destructor}}
}

void singleCast() {
  NVDerived *sc = new NVDerived();
  NonVirtual *sc2 = reinterpret_cast<NonVirtual*>(sc); // expected-note{{Casting from 'NVDerived' to 'NonVirtual' here}}
  delete sc2; // expected-warning{{Destruction of a polymorphic object with no virtual destructor}}
  // expected-note@-1{{Destruction of a polymorphic object with no virtual destructor}}
}

void doubleCast() {
  NonVirtual *dd = new NVDerived(); // expected-note {{Casting from 'NVDerived' to 'NonVirtual' here}}
  NVDerived *dd2 = reinterpret_cast<NVDerived*>(dd); // expected-note {{Casting from 'NonVirtual' to 'NVDerived' here}}
  dd = reinterpret_cast<NonVirtual*>(dd2); // expected-note {{Casting from 'NVDerived' to 'NonVirtual' here}}
  delete dd; // expected-warning {{Destruction of a polymorphic object with no virtual destructor}}
  // expected-note@-1{{Destruction of a polymorphic object with no virtual destructor}}
}

void implicitNV() {
  ImplicitNV *invd = new ImplicitNVDerived(); // expected-note{{Casting from 'ImplicitNVDerived' to 'ImplicitNV' here}}
  delete invd; // expected-warning{{Destruction of a polymorphic object with no virtual destructor}}
  // expected-note@-1{{Destruction of a polymorphic object with no virtual destructor}}
}

void doubleDecl() {
  ImplicitNV *dd1, *dd2;
  dd1 = new ImplicitNVDerived(); // expected-note{{Casting from 'ImplicitNVDerived' to 'ImplicitNV' here}}
  delete dd1; // expected-warning{{Destruction of a polymorphic object with no virtual destructor}}
  // expected-note@-1{{Destruction of a polymorphic object with no virtual destructor}}
}

void virtualBase() {
  Virtual *vb = new VDerived();
  delete vb; // no-warning
}

void notDerived() {
  NonVirtual *nd = new NonVirtual();
  delete nd; // no-warning
}

void notDerivedArr() {
  NonVirtual *nda = new NonVirtual[3];
  delete[] nda; // no-warning
}

void cast() {
  NonVirtual *c = new NVDerived();
  delete reinterpret_cast<NVDerived*>(c); // no-warning
}

void deleteThroughFunction2() {
  NonVirtual *dtf2 = new NVDerived();
  sinkCast(dtf2); // no-warning
}

void deleteThroughFunction3() {
  NVDerived *dtf3;
  dtf3 = new NVDerived();
  sinkParamCast(dtf3); // no-warning
}

void stackVar() {
  NonVirtual sv2;
  delete &sv2; // no-warning
}

// Deleting a polymorphic object with a non-virtual dtor
// is not a problem if it is referenced by its precise type.

void preciseType() {
  NVDerived *pt = new NVDerived();
  delete pt; // no-warning
}

void privateDtor() {
  Base *pd = new PrivateDtor();
  pd->destroy(); // no-warning
}
