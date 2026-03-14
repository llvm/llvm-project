int main() {
  return 0;
}

void function1(int *p) {
  if (!p) {
    *p = 7; // This will emit a null pointer diagnostic.
  }
}
