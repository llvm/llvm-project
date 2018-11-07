void pause() {}

int foo(int a) {
  int vla[a];

  for (int i = 0; i < a; ++i)
    vla[i] = a-i;

  pause(); // break here
  return vla[a-1];
}

int main (void) {
  return foo(2) + foo(4);
}
