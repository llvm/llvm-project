// RUN: %clang -g -S %s -o - | FileCheck %s
// Test to check presence of debug info for byval parameter.
// Radar 8350436.
class DAG {
public:
  int i;
  int j;
};

class EVT {
public:
  int a;
  int b;
  int c;
};

class VAL {
public:
  int x;
  int y;
};
void foo(EVT e);
EVT bar();

void get(int *i, unsigned dl, VAL v, VAL *p, unsigned n, EVT missing_arg) {
//CHECK: .asciz "missing_arg"
  EVT e = bar();
  if (dl == n)
    foo(missing_arg);
}

