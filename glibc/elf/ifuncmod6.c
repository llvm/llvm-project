/* Test STT_GNU_IFUNC symbol reference in a shared library.  */

extern int foo (void);

typedef int (*foo_p) (void);

extern foo_p foo_ptr;

foo_p
get_foo_p (void)
{
  return foo_ptr;
}

int
call_foo (void)
{
  return foo ();
}
