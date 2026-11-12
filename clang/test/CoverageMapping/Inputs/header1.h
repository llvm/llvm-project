#ifndef HEADER1_H
#define HEADER1_H

inline void func(int i) {
  int x = 0;
  if (i == 0) {
    x = 1;
  } else {
    x = 2;
  }
}
static void static_func(int j) {
  int x = 0;
  if (j == x) {
    x = !j;
  } else {
    x = 42;
  }
  j = x * j;
}
static void static_func2(int j) {
  int x = 0;
  if (j == x) {
    x = !j;
  } else {
    x = 42;
  }
  j = x * j;
}

#endif // HEADER1_H
