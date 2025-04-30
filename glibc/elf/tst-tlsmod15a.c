extern int nonexistent_dummy_var;
int *
foo (void)
{
  return &nonexistent_dummy_var;
}
