// RUN: %clang_cc1 -std=c23 %s -emit-llvm -o -
// RUN: %clang_cc1 -std=c23 %s -emit-llvm -o - -fexperimental-new-constant-interpreter

enum e : bool { b = true };
void foo ()
{
  enum e e1;
  e1 = (bool) nullptr;
}
