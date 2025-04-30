extern int circlemod1 (void);
extern int circlemod2 (void);

int
circlemod3 (void)
{
  return 3;
}

int
circlemod3a (void)
{
  return circlemod1 () + circlemod2 ();
}
