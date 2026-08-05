// depth(3): fully flatten a three-deep perfect nest (OpenMP 6.1).
extern "C" void body(int, int, int);
void t(int n, int m, int p) {
#pragma omp flatten depth(3)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      for (int k = 0; k < p; ++k)
        body(i, j, k);
}
