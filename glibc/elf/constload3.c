extern int baz (void);
extern int bar (void);

int
bar (void)
{
  return -21 + baz ();
}
