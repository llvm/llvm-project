int main(void) {
  int i = 22;

  __gcov_flush();

  i = 42;

  asm("int $3");

  i = 84;

  return 0;
}
