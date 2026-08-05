// Range-for (CXXForRangeStmt) nest + a function template instantiated with a
// dependent depth(K). Exercises the __begin/__end pre-inits and the deferred
// (dependent-context) desugaring path.
#include <stdlibc++.h>
extern "C" void body(int, int);

void ranges(std::vector<int> &a, std::vector<int> &b) {
#pragma omp flatten
  for (int i : a)
    for (int j : b)
      body(i, j);
}

template <int K> void tmpl(int n, int m) {
#pragma omp flatten depth(K)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      body(i, j);
}
template void tmpl<2>(int, int);
