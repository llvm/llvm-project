int main(void)
{
  int var;

  int a = 1;
  if (a) {
    var++;
  }

  if (a) {}

  int b = 0;
  if (b) {
    var++;
  }

  if (b) {}

  return 0;
}
