extern int bar (void);
extern int baz (void);

int
bar (void)
{
  return 32 + baz ();
}
