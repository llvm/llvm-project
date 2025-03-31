// RUN: %clang_cc1 -triple s390x-ibm-zos %s -fsyntax-only -verify

typedef int _Export ty;
ty x;
int f(int _Export x);
static int _Export s;
struct S {
  int _Export nonstaticdatamember;
};
void g() {
  int _Export automatic;
}
