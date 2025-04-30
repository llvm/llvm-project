__thread int a;

int f(void)
{
  return ++a;
}
