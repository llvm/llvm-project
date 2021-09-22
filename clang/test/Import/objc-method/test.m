// UNSUPPORTED: -zos, -aix
// RUN: clang-import-test -x objective-c++ -import %S/Inputs/S.m -expression %s
void expr() {
  C *c;
  int i = [c m];
}
