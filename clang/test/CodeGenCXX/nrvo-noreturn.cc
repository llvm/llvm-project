// RUN: %clang_cc1 -emit-llvm-only %s
// PR9178

void abort() __attribute__((__noreturn__));
struct CoinModelLink {
  CoinModelLink();
  ~CoinModelLink();
};
class CoinModel {
  CoinModelLink firstInQuadraticColumn();
};
CoinModelLink CoinModel::firstInQuadraticColumn() {
  abort();
  CoinModelLink x;
  return x;
}

