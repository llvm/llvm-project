// RUN: clang-import-test -import %S/Inputs/T.cpp -expression %s
// XFAIL: *
void expr() {
  A<int>::B b1;
  A<bool>::B b2;
  b1.f + b2.g;
}
