#include <stdlib.h>

struct A
{
  char a;
  int b;
  long long c;
};

extern __thread struct A a1, a2, a3, a4;
extern struct A *f1a (void);
extern struct A *f2a (void);
extern struct A *f3a (void);
extern struct A *f4a (void);
extern struct A *f5a (void);
extern struct A *f6a (void);
extern struct A *f7a (void);
extern struct A *f8a (void);
extern struct A *f9a (void);
extern struct A *f10a (void);
extern int f1b (void);
extern int f2b (void);
extern int f3b (void);
extern int f4b (void);
extern int f5b (void);
extern int f6b (void);
extern int f7b (void);
extern int f8b (void);
extern int f9b (void);
extern int f10b (void);
extern void check1 (void);
extern void check2 (void);
