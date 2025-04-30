extern int call_me (void);
extern int bar (void);

int
bar (void)
{
  return call_me ();
}
