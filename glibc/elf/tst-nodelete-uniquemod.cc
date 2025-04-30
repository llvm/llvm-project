extern int not_exist (void);

inline int make_unique (void)
{
  /* Static variables in inline functions and classes
     generate STB_GNU_UNIQUE symbols.  */
  static int unique;
  return ++unique;
}

int foo (void)
{
  return make_unique () + not_exist ();
}
