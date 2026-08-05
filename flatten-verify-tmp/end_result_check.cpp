// End-result (semantic) verification: flatten must preserve BOTH the set of
// iterations and their row-major visitation ORDER. We record the sequence of
// visited tuples from a flattened nest and compare it against the sequence from
// an identical non-flattened reference nest. Exit code 0 == identical.
//
// flatten is a pure frontend AST desugaring to an ordinary loop, so this runs
// as a normal program (build with: clang++ -fopenmp -fopenmp-version=61).
#include <cassert>
#include <cstdio>
#include <vector>

using Seq = std::vector<long>;

static void rec(Seq &s, int i, int j, int k = -1) {
  s.push_back(i);
  s.push_back(j);
  if (k >= 0)
    s.push_back(k);
}

int main() {
  const int n = 3, m = 4, p = 2;

  // --- depth(2) default over 2 loops ---
  {
    Seq flat, ref;
#pragma omp flatten
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j)
        rec(flat, i, j);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j)
        rec(ref, i, j);
    assert(flat == ref && "depth(2) order/set mismatch");
  }

  // --- depth(3) full flatten over 3 loops ---
  {
    Seq flat, ref;
#pragma omp flatten depth(3)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j)
        for (int k = 0; k < p; ++k)
          rec(flat, i, j, k);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j)
        for (int k = 0; k < p; ++k)
          rec(ref, i, j, k);
    assert(flat == ref && "depth(3) order/set mismatch");
  }

  // --- depth(2) partial: inner k-loop stays intact ---
  {
    Seq flat, ref;
#pragma omp flatten depth(2)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j)
        for (int k = 0; k < p; ++k)
          rec(flat, i, j, k);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j)
        for (int k = 0; k < p; ++k)
          rec(ref, i, j, k);
    assert(flat == ref && "depth(2)-partial order/set mismatch");
  }

  // --- non-unit step / non-zero start must survive ---
  {
    Seq flat, ref;
#pragma omp flatten
    for (int i = 5; i < 12; i += 2)
      for (int j = 1; j < 7; j += 3)
        rec(flat, i, j);
    for (int i = 5; i < 12; i += 2)
      for (int j = 1; j < 7; j += 3)
        rec(ref, i, j);
    assert(flat == ref && "stride/offset mismatch");
  }

  std::puts("flatten end-result preserved: OK");
  return 0;
}
