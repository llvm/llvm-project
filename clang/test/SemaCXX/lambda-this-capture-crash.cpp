// RUN: %clang_cc1 -std=c++23 -verify -fsyntax-only %s

// Test case for ensuring lambda *this capture with explicit this parameter doesn't crash
// This reproduces a crash that occurs in some clang distributions (e.g., Apple clang 17.0.0)
// when combining lambda capture by copy of *this with explicit this parameters

struct S {
  int x;
  auto byval() {
    // This combination should not crash the compiler
    return [*this](this auto) { return this->x; }; // expected-no-diagnostics
  }
};

// Variation with explicit type and parameter name
struct S1 {
  int x;
  auto byval() {
    return [*this](this auto&& self) { return this->x; }; // expected-no-diagnostics
  }
};

// Using captured member without explicit this->
struct S2 {
  int x;
  auto byval() {
    return [*this](this auto&& self) { return x; }; // expected-no-diagnostics
  }
};

// More complex case with multiple members and parameters
struct S3 {
  int x;
  int y;
  auto complex() {
    return [*this](this auto&& self, int z) { 
      return this->x + this->y + z; 
    }; // expected-no-diagnostics
  }
};

// Nested lambda case
struct S4 {
  int x;
  auto nested() {
    return [*this](this auto&& self) {
      return [*this](this auto&& inner) { return this->x; };
    }; // expected-no-diagnostics
  }
};

// Test that the code actually compiles and works semantically
constexpr int test() {
  S s{ 42 };
  auto lambda = s.byval();
  return lambda(); // Should return 42
}

static_assert(test() == 42); 