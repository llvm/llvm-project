extern int var_in_mod4;
extern int *addr (void);

int *
addr (void)
{
  return &var_in_mod4;
}
