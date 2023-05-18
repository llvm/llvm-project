#ifndef COMMON_H
#define COMMON_H

extern void ext(int (&)[5]);

void func(int t) {
  int ints[5];
  for (unsigned i = 0; i < 5; ++i) {
    ints[i] = t;
  }

  int *i = 0;

  ext(ints);
}

#endif // COMMON_H
