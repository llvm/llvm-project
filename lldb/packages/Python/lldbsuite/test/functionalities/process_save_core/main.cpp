int global = 42;

int
bar(int x)
{
  int y = 4*x + global;
  return y;
}

int
foo(int x)
{
  int y = 2*bar(3*x);
  return y;
}

int
main()
{
  return 0 * foo(1);
}
