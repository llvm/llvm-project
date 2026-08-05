// depth(2) on a three-deep nest: only the outer two loops are combined; the
// innermost loop must remain as ordinary body of the flattened loop.
extern "C" void body(int, int, int);
void t(int n, int m, int p) {
#pragma omp flatten depth(2)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      for (int k = 0; k < p; ++k)
        body(i, j, k);
}
