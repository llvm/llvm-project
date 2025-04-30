void
__attribute__((constructor))
xxx (void)
{
  extern void baz (void);
  baz ();
}
