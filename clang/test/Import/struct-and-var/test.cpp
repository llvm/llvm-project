// RUN: clang-import-test --import %S/Inputs/S1.cpp --import %S/Inputs/S2.cpp -expression %s
void expr() {
  struct F f;
  int x = f.a;
}
