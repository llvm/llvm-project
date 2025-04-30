extern int foo (void);
extern int bar (void);

int
bar (void)
{
  return foo ();
}
