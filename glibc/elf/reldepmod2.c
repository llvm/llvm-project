extern int foo (void);
extern int call_me (void);

int
call_me (void)
{
  return foo () - 42;
}
