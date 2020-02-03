void function2(int *o) {
  if (!o) {
    *o = 7; // This will emit a null pointer diagnostic.
  }
}
