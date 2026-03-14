// RUN: clang-import-test -import %S/Inputs/F.cpp -expression %s
void expr() {
  X X1;
  Y Y1 = X1;
}
