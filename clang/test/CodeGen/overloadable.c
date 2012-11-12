// RUN: %clang_cc1 -emit-llvm %s -o - | grep _Z1fPA10_1X
int __attribute__((overloadable)) f(int x) { return x; }
float __attribute__((overloadable)) f(float x) { return x; }
double __attribute__((overloadable)) f(double x) { return x; }
double _Complex __attribute__((overloadable)) f(double _Complex x) { return x; }
typedef short v4hi __attribute__ ((__vector_size__ (8)));
v4hi __attribute__((overloadable)) f(v4hi x) { return x; }

struct X { };
void  __attribute__((overloadable)) f(struct X (*ptr)[10]) { }

void __attribute__((overloadable)) f(int x, int y, ...) { }

int main() {
  int iv = 17;
  float fv = 3.0f;
  double dv = 4.0;
  double _Complex cdv;
  v4hi vv;

  iv = f(iv);
  fv = f(fv);
  dv = f(dv);
  cdv = f(cdv);
  vv = f(vv);
}
