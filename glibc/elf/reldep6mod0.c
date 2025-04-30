int bar (void);
extern void free (void *);

int bar (void)
{
  free (0);
  return 40;
}
