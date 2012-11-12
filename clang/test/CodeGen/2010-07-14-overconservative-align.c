// RUN: %clang_cc1 %s -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s
// PR 5995
struct s {
  int word;
  struct {
    int filler __attribute__ ((aligned (8)));
  };
};

void func (struct s *s)
{
  // CHECK: load %struct.s**{{.*}}align 8
  s->word = 0;
}
