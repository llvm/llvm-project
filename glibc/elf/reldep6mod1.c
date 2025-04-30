int foo (void);
int baz (void);
extern int weak (void);
asm (".weak weak");

int foo (void)
{
  return 20;
}

int baz (void)
{
  return weak () + 1;
}
