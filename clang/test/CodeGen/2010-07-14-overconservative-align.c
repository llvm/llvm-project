// RUN: %clang_cc1 %s -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir %s -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s %}
// PR 5995
struct s {
  int word;
  struct {
    int filler __attribute__ ((aligned (8)));
  };
};

void func (struct s *s)
{
  // CHECK: load ptr, ptr{{.*}}align 8
  s->word = 0;
}
