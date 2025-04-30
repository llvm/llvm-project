extern int bar (void);
extern int foo (void);

int
foo (void)
{
  return 10 + bar ();
}
