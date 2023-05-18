// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// expected-no-diagnostics

class Trivial { int n; void f(); };
class NonTrivial1 { NonTrivial1(const NonTrivial1 &); };
class NonTrivial2 { NonTrivial2(NonTrivial2 &&); };
class NonTrivial3 { NonTrivial3 operator=(const NonTrivial3 &); };
class NonTrivial4 { NonTrivial4 operator=(NonTrivial4 &&); };
class NonTrivial5 { ~NonTrivial5(); };

static_assert(__is_trivial(Trivial), "Trivial is not trivial");
static_assert(!__is_trivial(NonTrivial1), "NonTrivial1 is trivial");
static_assert(!__is_trivial(NonTrivial2), "NonTrivial2 is trivial");
static_assert(!__is_trivial(NonTrivial3), "NonTrivial3 is trivial");
static_assert(!__is_trivial(NonTrivial4), "NonTrivial4 is trivial");
static_assert(!__is_trivial(NonTrivial5), "NonTrivial5 is trivial");

struct Trivial2 {
  Trivial2() = default;
  Trivial2(const Trivial2 &) = default;
  Trivial2(Trivial2 &&) = default;
  Trivial2 &operator=(const Trivial2 &) = default;
  Trivial2 &operator=(Trivial2 &&) = default;
  ~Trivial2() = default;
};

class NonTrivial6 { ~NonTrivial6(); };

NonTrivial6::~NonTrivial6() = default;

static_assert(!__is_trivial(NonTrivial6), "NonTrivial6 is trivial");
