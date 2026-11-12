// RUN: clang-import-test -import %S/Inputs/N1.cpp -import %S/Inputs/N2.cpp -import %S/Inputs/N3.cpp -expression %s
void expr() {
  N::S s;
  N::T t;
  N::U u;
  int d = s.a + t.b + u.c;
}
