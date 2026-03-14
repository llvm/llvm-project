// RUN: clang-import-test -import %S/Inputs/F.cpp -expression %s
void expr() {
  f(2);
}