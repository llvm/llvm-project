// RUN: %clang_cc1 -fopenmp -emit-llvm -std=c++20 %s -o -
struct Iter {
  const int *Ptr;

  bool operator!=(const Iter &Other) const { return Ptr != Other.Ptr; }
  void operator++() { ++Ptr; }
  const int &operator*() const { return *Ptr; }
  long operator-(const Iter &Other) const { return Ptr - Other.Ptr; }
  void operator+=(long N) { Ptr += N; }
};

struct Range {
  int Data[4];
};

Iter begin(const Range &R) { return {R.Data}; }
Iter end(const Range &R) { return {R.Data + 4}; }

template <typename T>
void foo() {
  Range R;

  auto lambda = [R]() {
#pragma omp for
    for (auto x : R)
      ;
  };

  lambda();
}

template void foo<int>();
