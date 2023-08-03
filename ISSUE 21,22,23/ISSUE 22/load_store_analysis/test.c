struct S {
  int a;
  struct B {
    int b;
  } c;
};

int main() {
  struct S s;
  s.a = 5;
  s.c.b = 10;

  int* p = &s.a;
  int* q = &s.c.b;

  int x = *p;
  int y = *q;

  printf("x = %d\n", x);
  printf("y = %d\n", y);

  return 0;
}
