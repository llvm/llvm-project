template <typename T, typename U> concept convertible_to = true;
template <typename T, typename U> concept same_as = true;
template <typename T> concept integral = true;

template <typename A, typename B>
concept W = requires(A a, B b) {
  { b.www } noexcept -> integral;
};

template <typename T> concept X = requires(T t) {
  t.xxx(42);
  typename T::xxx_t;
  T::xyz::member;
};

template <typename T, typename U>
concept Y = requires(T t, U u) { t.yyy(u); };

template <typename T>
concept Z = requires(T t) {
  { t.zzz() } -> same_as<int>;
  requires W<int, T>;
};

// Concept constraints in all three slots require X, Y, Z, and ad-hoc stuff.
template <X T>
requires Y<T, int> && requires(T *t) { { t->aaa() } -> convertible_to<double>; }
void foo(T t) requires Z<T> || requires(T &t) { t.bbb(); t->bb(); } {
  t.x;
  t->x;
  T::x;

  // RUN: %clang_cc1 -std=c++2a -code-completion-with-fixits -code-completion-at=%s:29:5 %s \
  // RUN: | FileCheck %s -check-prefix=DOT -implicit-check-not=xxx_t
  // DOT: Pattern : [#convertible_to<double>#]aaa()
  // DOT: Pattern : bb() (requires fix-it: {{.*}} to "->")
  // DOT: Pattern : bbb()
  // DOT: Pattern : [#integral#]www
  // DOT: Pattern : xxx(<#int#>)
  // FIXME: it would be nice to have int instead of U here.
  // DOT: Pattern : yyy(<#U#>)
  // DOT: Pattern : [#int#]zzz()

  // RUN: %clang_cc1 -std=c++2a -code-completion-with-fixits -code-completion-at=%s:30:6 %s \
  // RUN: | FileCheck %s -check-prefix=ARROW -implicit-check-not=xxx_t
  // ARROW: Pattern : [#convertible_to<double>#]aaa() (requires fix-it: {{.*}} to ".")
  // ARROW: Pattern : bb()
  // ARROW: Pattern : bbb() (requires fix-it
  // ARROW: Pattern : [#integral#]www (requires fix-it
  // ARROW: Pattern : xxx(<#int#>) (requires fix-it
  // ARROW: Pattern : yyy(<#U#>) (requires fix-it
  // ARROW: Pattern : [#int#]zzz() (requires fix-it

  // RUN: %clang_cc1 -std=c++2a -code-completion-with-fixits -code-completion-at=%s:31:6 %s \
  // RUN: | FileCheck %s -check-prefix=COLONS -implicit-check-not=yyy
  // COLONS: Pattern : xxx_t
  // COLONS: Pattern : xyz
}

