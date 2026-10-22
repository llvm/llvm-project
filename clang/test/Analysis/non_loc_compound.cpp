// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=alpha.cplusplus.InvalidatedIterator \
// RUN:   -analyzer-config aggressive-binary-operation-simplification=true \
// RUN:   2>&1

struct node {};
struct prop : node {};
struct bitvec : node {
  prop operator==(bitvec) { return prop(); }
  bitvec extend(); // { return *this; }
};
void convert() {
  bitvec input;
  bitvec output(input.extend());
  output == input;
}
