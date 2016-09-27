// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.core.CastToStruct,core -verify %s

struct AB {
  int A;
  int B;
};

struct ABC {
  int A;
  int B;
  int C;
};

struct Base {
  Base() : A(0), B(0) {}
  virtual ~Base() {}

  int A;
  int B;
};

struct Derived : public Base {
  Derived() : Base(), C(0) {}
  int C;
};

void structToStruct(struct AB *P) {
  struct AB Ab;
  struct ABC *Abc;
  Abc = (struct ABC *)&Ab; // expected-warning {{Casting data to a larger structure type and accessing a field can lead to memory access errors or data corruption}}
  Abc = (struct ABC *)P; // No warning; It is not known what data P points at.
  Abc = (struct ABC *)&*P;

  // Don't warn when the cast is not widening.
  P = (struct AB *)&Ab; // struct AB * => struct AB *
  struct ABC Abc2;
  P = (struct AB *)&Abc2; // struct ABC * => struct AB *

  // True negatives when casting from Base to Derived.
  Derived D1, *D2;
  Base &B1 = D1;
  D2 = (Derived *)&B1;
  D2 = dynamic_cast<Derived *>(&B1);
  D2 = static_cast<Derived *>(&B1);

  // True positives when casting from Base to Derived.
  Base B2;
  D2 = (Derived *)&B2;// expected-warning {{Casting data to a larger structure type and accessing a field can lead to memory access errors or data corruption}}
  D2 = dynamic_cast<Derived *>(&B2);// expected-warning {{Casting data to a larger structure type and accessing a field can lead to memory access errors or data corruption}}
  D2 = static_cast<Derived *>(&B2);// expected-warning {{Casting data to a larger structure type and accessing a field can lead to memory access errors or data corruption}}

  // False negatives, cast from Base to Derived. With path sensitive analysis
  // these false negatives could be fixed.
  Base *B3 = &B2;
  D2 = (Derived *)B3;
  D2 = dynamic_cast<Derived *>(B3);
  D2 = static_cast<Derived *>(B3);
}

void intToStruct(int *P) {
  struct ABC *Abc;
  Abc = (struct ABC *)P; // expected-warning {{Casting a non-structure type to a structure type and accessing a field can lead to memory access errors or data corruption}}

  // Cast from void *.
  void *VP = P;
  Abc = (struct ABC *)VP;
}
