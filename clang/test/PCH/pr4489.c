// RUN: %clang -x c-header -o %t.pch %s
// RUN: echo > %t.empty.c
// RUN: %clang -include %t -x c %t.empty.c -emit-llvm -S -o -

// PR 4489: Crash with PCH
// PR 4492: Crash with PCH (round two)
// PR 4509: Crash with PCH (round three)
typedef struct _IO_FILE FILE;
extern int fprintf (struct _IO_FILE *__restrict __stream,
                    __const char *__restrict __format, ...);

int x(void)
{
  switch (1) {
    case 2: ;
      int y = 0;
  }
}

void y(void) {
  extern char z;
  fprintf (0, "a");
}

struct y0 { int i; } y0[1] = {};

void x0(void)
{
  extern char z0;
  fprintf (0, "a");
}

void x1(void)
{
  fprintf (0, "asdf");
}

void y1(void)
{
  extern char e;
  fprintf (0, "asdf");
}
