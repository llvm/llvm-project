// RUN: %clang_cc1 -std=c++23 -verify -fsyntax-only %s

// Test case for crash when combining lambda *this capture with explicit this parameter
// This should not crash the compiler

struct S {
  int x;
  auto byval() {
    return [*this](this auto) { return this->x; }; // expected-no-diagnostics
  }
};

// Variation with explicit type
struct S1 {
  int x;
  auto byval() {
    return [*this](this auto&& self) { return this->x; }; // expected-no-diagnostics
  }
};

// Using captured member without this->
struct S2 {
  int x;
  auto byval() {
    return [*this](this auto&& self) { return x; }; // expected-no-diagnostics
  }
};

// More complex case with multiple members
struct S3 {
  int x;
  int y;
  auto complex() {
    return [*this](this auto&& self, int z) { 
      return this->x + this->y + z; 
    }; // expected-no-diagnostics
  }
};

// Test that the code actually works
int test() {
  S s{ 42 };
  auto lambda = s.byval();
  return lambda(); // Should return 42
} 