int foo (void) __attribute__ ((weak));
int
foo (void)
{
  return 2;
}

int
mod2_bar (void)
{
  return foo ();
}
