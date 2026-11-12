// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
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

void GH3206(char *a, char *b) {
  if (a != b) {
    if (a != nullptr || b == nullptr) {
      // no-warning: 'a' is checked against null and even if 'b' is null the
      // outer if statement rules out entering with both 'a' and 'b' being null.
      a[0] = '\0';
    }
  }
}
