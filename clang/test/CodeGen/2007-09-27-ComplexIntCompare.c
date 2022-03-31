// RUN: %clang_cc1 -emit-llvm %s -o -
// PR1708

void __attribute__((noreturn)) abort(void);

struct s { _Complex unsigned short x; };
struct s gs = { 100 + 200i };
struct s __attribute__((noinline)) foo (void) { return gs; }

int main (void)
{
  if (foo ().x != gs.x)
    abort ();
  exit (0);
}
