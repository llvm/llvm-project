// Static analyzer invocation on split loop (no crash).
// RUN: %clang -target x86_64-unknown-linux-gnu --analyze -fopenmp -fopenmp-version=60 %s -o %t.plist

void g(int);

void f(int n) {
#pragma omp split counts(2, omp_fill)
  for (int i = 0; i < n; ++i)
    g(i);
}
