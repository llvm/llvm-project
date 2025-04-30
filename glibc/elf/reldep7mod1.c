int foo (void) __attribute__ ((weak));
int
foo (void)
{
  return 1;
}

int
mod1_bar (void)
{
  return foo ();
}
