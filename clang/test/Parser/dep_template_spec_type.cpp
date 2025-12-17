// RUN: seq 100 | xargs -Ifoo %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
// This is a regression test for a non-deterministic stack-overflow.

template <typename C, typename S1, int rbits>
typename C::A Bar(const S1& x, const C& c = C()) {
    using T = typename C::A;
    T result;

    using PreC = typename C::template boop<T::p + rbits>;
    using ExactC = typename C::template bap<PreC::p + 2>;

    using D = typename ExactC::A;

    return result;
}
