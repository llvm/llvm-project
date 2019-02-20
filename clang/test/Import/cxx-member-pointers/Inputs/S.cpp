struct S {
  int i;
};

int S::*iptr() {
  return &S::i;
}
