int var_in_mod4 = 99;

static void
__attribute__ ((__destructor__))
destr (void)
{
  extern int fini_ran;
  fini_ran = 1;
}
