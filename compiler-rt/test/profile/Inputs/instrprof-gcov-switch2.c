int main(void)
{
  int i = 22;

  switch (i) {
    case 7:
      break;

    case 22:
      i = 7;

    case 42:
      i = 22;
      break;
  }

  return 0;
}
