// RUN: %clang_cc1 -target-feature +altivec -fsyntax-only %s -triple powerpc64-ibm-aix-xcoff
// RUN: %clang_cc1 -target-feature +altivec -fsyntax-only %s -triple powerpc64le-ibm-linux-gnu
// RUN: %clang_cc1 -target-feature +altivec -fsyntax-only %s -triple powerpc64-linux-gnu
// RUN: %clang_cc1 -target-feature +altivec -fsyntax-only %s -triple powerpc-ibm-aix-xcoff
// RUN: %clang_cc1 -target-feature +altivec -fsyntax-only %s -triple powerpc-linux-gnu

int vector();

void test() {
  vector unsigned int v = {0};
}
