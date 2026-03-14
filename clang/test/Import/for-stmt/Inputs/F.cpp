void f() {
  for (;;)
    ;
  for (int i = 0;;)
    continue;
  for (; bool j = false;)
    continue;
  for (int i = 0; i != 0; ++i) {
    i++;
  }
}
