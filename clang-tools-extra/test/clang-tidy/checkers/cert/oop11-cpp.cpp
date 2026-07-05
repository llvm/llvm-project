// RUN: %check_clang_tidy %s cert-oop11-cpp %t

struct B {
  B(B&&) noexcept = default;

  B(const B &) = default;
  B& operator=(const B&) = default;
  ~B() {}
};

struct D {
  B b;

  // CHECK-MESSAGES: :[[@LINE+1]]:14: warning: move constructor initializes class member by calling a copy constructor [cert-oop11-cpp]
  D(D &&d) : b(d.b) {}

  // This should not produce a diagnostic because it is not covered under
  // the CERT guideline for OOP11-CPP. However, this will produce a diagnostic
  // under performance-move-constructor-init.
  D(B b) : b(b) {}
};
