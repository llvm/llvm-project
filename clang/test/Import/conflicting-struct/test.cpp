// RUN: clang-import-test --import %S/Inputs/S1.cpp --import %S/Inputs/S2.cpp -expression %s
void expr() {
  S MyS;
  T MyT;
  MyS.a = 3;
  MyT.u.b = 2;
}
