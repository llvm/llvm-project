// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

// rdar://problem/56586853
// expected-no-diagnostics

struct Data {
  int x;
  Data *data;
};

int compare(Data &a, Data &b) {
  Data *aData = a.data;
  Data *bData = b.data;

  // Covers the cases where both pointers are null as well as both pointing to the same buffer.
  if (aData == bData)
    return 0;

  if (aData && !bData)
    return 1;

  if (!aData && bData)
    return -1;

  return compare(*aData, *bData); // no-warning
}
