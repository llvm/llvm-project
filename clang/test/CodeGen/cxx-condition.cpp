// RUN: %clang_cc1 -emit-llvm %s -o %t

void f() {
  int a;
  if (int x=a) ++a; else a=x;
  while (int x=a) ++a;
  for (; int x=a; --a) ;
  switch (int x=0) { }
}
