extern int exfunc(int a);

static inline int hfunc1()
{
  return exfunc(1);
}
