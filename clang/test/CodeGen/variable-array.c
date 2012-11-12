// RUN: %clang_cc1 -emit-llvm < %s | grep puts | count 4

// PR3248
int a(int x)
{
  int (*y)[x];
  return sizeof(*(puts("asdf"),y));
}

// PR3247
int b() {
  return sizeof(*(char(*)[puts("asdf")])0);
}

// PR3247
int c() {
  static int (*y)[puts("asdf")];
  return sizeof(*y);
}
