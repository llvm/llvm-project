// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

/* Make sure the frontend is correctly marking static stuff as internal! */

int X;
static int Y = 12;

static void foo(int Z) {
  Y = Z;
}

void *test(void) {
  foo(12);
  return &Y;
}
