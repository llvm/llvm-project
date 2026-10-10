// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

int gv = 5; // expected-note 4 {{declared here}}
constexpr int cgv = 5;

struct W {
  int value;
  consteval W(int x) : value(x) {}
};

consteval int id(int x) { return x; }

struct S {
  int member = [captured = W(gv).value] { return captured; }(); // expected-error {{call to consteval function 'W::W' is not a constant expression}} \
  // expected-note {{declared here}} \
  // expected-note {{read of non-const variable 'gv' is not allowed in a constant expression}}
};

struct F {
  int member = [captured = id(gv)] { return captured; }(); // expected-error {{call to consteval function 'id' is not a constant expression}} \
  // expected-note {{declared here}} \
  // expected-note {{read of non-const variable 'gv' is not allowed in a constant expression}}
};

struct MixedThisFirst {
  int member = [this, captured = W(gv).value] { return captured; }(); // expected-error {{call to consteval function 'W::W' is not a constant expression}} \
  // expected-note {{declared here}} \
  // expected-note {{read of non-const variable 'gv' is not allowed in a constant expression}}
};

struct MixedThisLast {
  int member = [captured = id(gv), this] { return captured; }(); // expected-error {{call to consteval function 'id' is not a constant expression}} \
  // expected-note {{declared here}} \
  // expected-note {{read of non-const variable 'gv' is not allowed in a constant expression}}
};

struct Valid {
  int member = [captured = W(cgv).value] { return captured; }();
  int function = [captured = id(cgv)] { return captured; }();

  int mixed_ctor =
      [this, captured = W(cgv).value] { return captured; }();
  int mixed_function =
      [captured = id(cgv), this] { return captured; }();
};

static_assert(Valid{}.member == 5);
static_assert(Valid{}.function == 5);
static_assert(Valid{}.mixed_ctor == 5);
static_assert(Valid{}.mixed_function == 5);

int main() {
  return S{}.member + F{}.member + MixedThisFirst{}.member + MixedThisLast{}.member; // expected-note 4 {{in the default initializer of 'member'}}
}

namespace consteval_copy_capture {
struct Copyable {
  int value;
  constexpr Copyable(int x) : value(x) {}
  consteval Copyable(const Copyable &other) : value(other.value) {} // expected-note {{read of non-constexpr variable 'runtime_source' is not allowed in a constant expression}} \
  // expected-note {{read of non-constexpr variable 'local_source' is not allowed in a constant expression}}
};

Copyable runtime_source{5}; // expected-note {{declared here}}
constexpr Copyable constant_source{5};

// Copying an existing lvalue forces the consteval copy constructor to run.
struct InvalidInitCopy {
  int member = [captured = runtime_source] { return captured.value; }(); // expected-error {{call to consteval function 'consteval_copy_capture::Copyable::Copyable' is not a constant expression}} \
  // expected-note {{declared here}} \
  // expected-note {{in call to 'Copyable(runtime_source)'}}
};

struct ValidInitCopy {
  int member = [captured = constant_source] { return captured.value; }();
};

static_assert(ValidInitCopy{}.member == 5);

int use_invalid_init_copy() {
  return InvalidInitCopy{}.member; // expected-note {{in the default initializer of 'member'}}
}

// Also cover ordinary by-value captures.
void invalid_regular_copy() {
  Copyable local_source{5}; // expected-note {{declared here}}
  auto lambda = [local_source] { return local_source.value; }; // expected-error {{call to consteval function 'consteval_copy_capture::Copyable::Copyable' is not a constant expression}} \
  // expected-note {{in call to 'Copyable(local_source)'}}
}

constexpr int valid_regular_copy() {
  constexpr Copyable local_source{5};
  return [local_source] { return local_source.value; }();
}

static_assert(valid_regular_copy() == 5);
} // namespace consteval_copy_capture
