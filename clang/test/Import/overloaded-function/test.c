// RUN: clang-import-test -import %S/Inputs/F1.c -import %S/Inputs/F2.c -expression %s
void expr() {
  f(2);
  f("world");
  S s;
  f(s);
}
