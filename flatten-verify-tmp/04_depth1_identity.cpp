// depth(1): identity transform. A single loop is "combined"; no mixed-radix
// decomposition (no srem), the flattened counter maps straight to the loop var.
extern "C" void body(int);
void t(int n) {
#pragma omp flatten depth(1)
  for (int i = 0; i < n; ++i)
    body(i);
}
