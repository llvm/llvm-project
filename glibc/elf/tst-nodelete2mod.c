/* Undefined symbol.  */
extern int not_exist (void);

int foo (void)
{
  return not_exist ();
}
