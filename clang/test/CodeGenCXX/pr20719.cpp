// RUN: %clang_cc1 -triple i686-windows-msvc -emit-llvm -std=c++11 -o - %s | FileCheck %s

// Make sure that we emit H's constructor twice: once with the first lambda
// inside of 'lep' and again with the second lambda inside of 'lep'.
// CHECK-DAG: @"\01??0?$H@V<lambda_1>@?0???$lep@X@@YAXXZ@@@QAE@XZ"
// CHECK-DAG: @"\01??0?$H@V<lambda_2>@?0???$lep@X@@YAXXZ@@@QAE@XZ"

template <typename>
struct H {
  H() {}
};

template <typename Fx>
int K_void(const Fx &) {
  H<Fx> callee;
  return 0;
}
template <typename Fx>
int K_int(const Fx &) {
  H<Fx> callee;
  return 0;
}

struct pair {
  pair(int, int);
};

struct E1;

template <typename>
void lep() {
  pair x(K_void([] {}), K_int([] {}));
}

auto z = lep<void>;
