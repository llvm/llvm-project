int main(void)
{
  int i = 22;

  switch (i) {
    case 7:
      break;

    case 22:
      i = 7;
      break;

    case 42:
      break;
  }

  return 0;
}
