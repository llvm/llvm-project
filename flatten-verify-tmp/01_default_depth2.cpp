// Default flatten: combine 2 perfectly nested loops (depth defaults to 2).
extern "C" void body(int, int);
void t(int n, int m) {
#pragma omp flatten
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      body(i, j);
}
