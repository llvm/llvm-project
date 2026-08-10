// RUN: %clang_cc1 -target-feature +altivec -fsyntax-only %s -triple powerpc64-ibm-aix-xcoff
// RUN: %clang_cc1 -target-feature +altivec -fsyntax-only %s -triple powerpc64le-ibm-linux-gnu
// RUN: %clang_cc1 -target-feature +altivec -fsyntax-only %s -triple powerpc64-linux-gnu
// RUN: %clang_cc1 -target-feature +altivec -fsyntax-only %s -triple powerpc-ibm-aix-xcoff
// RUN: %clang_cc1 -target-feature +altivec -fsyntax-only %s -triple powerpc-linux-gnu

template <typename T> class vector {
public:
  vector(int) {}
};

void f() {
  vector int v = {0};
  vector<int> vi = {0};
}
