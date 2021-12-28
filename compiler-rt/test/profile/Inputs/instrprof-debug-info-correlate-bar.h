int foo(int);

inline int bar(int a) {
  while (a > 100)
    a /= 2;
  return a;
}
