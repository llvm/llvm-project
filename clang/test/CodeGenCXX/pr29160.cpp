// RUN: %clang_cc1 -std=c++11 -triple i686-linux-gnu %s -o /dev/null -emit-llvm
//
// This test's failure mode is running ~forever. (For some value of "forever"
// that's greater than 25 minutes on my machine)

template <typename... Ts>
struct Foo {
  template <typename... T>
  static void ignore() {}
  Foo() { ignore<Ts...>(); }
};

struct Base {
  Base();
  ~Base();
};

#define STAMP(thiz, prev) using thiz = Foo< \
  prev, prev, prev, prev, prev, prev, prev, prev, prev, prev, prev, prev, \
  prev, prev, prev, prev, prev, prev, prev, prev, prev, prev, prev, prev, \
  prev, prev, prev, prev, prev, prev, prev, prev, prev, prev, prev, prev \
  >;
STAMP(A, Base);
STAMP(B, A);
STAMP(C, B);
STAMP(D, C);
STAMP(E, D);
STAMP(F, E);
STAMP(G, F);
STAMP(H, G);
STAMP(I, H);
STAMP(J, I);
STAMP(K, J);
STAMP(L, K);
STAMP(M, L);
STAMP(N, M);
STAMP(O, N);
STAMP(P, O);
STAMP(Q, P);

int main() { Q q; }
