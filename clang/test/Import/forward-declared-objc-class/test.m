// UNSUPPORTED: -zos, -aix
// RUN: clang-import-test -x objective-c++ -import %S/Inputs/S1.m --import %S/Inputs/S2.m --import %S/Inputs/S3.m -expression %s
void expr() {
  MyClass *c = [MyClass fromInteger:3];
  const int i = [c getInteger];
  const int j = c->j;
}
