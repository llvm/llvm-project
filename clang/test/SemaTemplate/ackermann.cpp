// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// template<unsigned M, unsigned N>
// struct Ackermann {
//   enum {
//     value = M ? (N ? Ackermann<M-1, Ackermann<M, N-1> >::value
//                    : Ackermann<M-1, 1>::value)
//               : N + 1
//   };
// };

template<unsigned M, unsigned N>
struct Ackermann {
 enum {
   value = Ackermann<M-1, Ackermann<M, N-1>::value >::value
 };
};

template<unsigned M> struct Ackermann<M, 0> {
 enum {
   value = Ackermann<M-1, 1>::value
 };
};

template<unsigned N> struct Ackermann<0, N> {
 enum {
   value = N + 1
 };
};

template<> struct Ackermann<0, 0> {
 enum {
   value = 1
 };
};

int g0[Ackermann<3, 4>::value == 125 ? 1 : -1];

