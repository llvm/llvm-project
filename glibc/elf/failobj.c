/* This function is supposed to not exist.  */
extern int xyzzy (int);

extern int foo (int);

int
foo (int a)
{
  return xyzzy (a);
}
