int foo(void) {
  int x;

  asm goto(""
           : "=r"(x)
           :
           :
           : indirect);
  x = 42;

indirect:
  return x;
}
