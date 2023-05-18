// RUN: %clang_cc1 -std=c++20 -x c++ %s -verify -fsyntax-only
// expected-no-diagnostics

template <typename T0>
concept HasMemberBegin = requires(T0 t) { t.begin(); };

struct GetBegin {
  template <HasMemberBegin T1>
  void operator()(T1);
};

GetBegin begin;

template <typename T2>
concept Concept = requires(T2 t) { begin(t); };

struct Subrange;

template <typename T3>
struct View {
  Subrange &getSubrange();

  operator bool()
    requires true;

  operator bool()
    requires requires { begin(getSubrange()); };

  void begin();
};

struct Subrange : View<void> {};
static_assert(Concept<Subrange>);
