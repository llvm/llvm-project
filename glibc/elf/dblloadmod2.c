extern int bar (void);
extern int baz (void);
extern int xyzzy (void);

int
baz (void)
{
  return -42;
}

int
xyzzy (void)
{
  return 10 + bar ();
}
