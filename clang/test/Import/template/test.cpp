// RUN: clang-import-test -import %S/Inputs/T.cpp -expression %s
void expr() {
  A<int>::B b;
}
