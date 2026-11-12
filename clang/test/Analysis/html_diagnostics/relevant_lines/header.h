#define deref(X) (*X)

char helper(
    char *out,
    int doDereference) {
  if (doDereference) {
    return deref(out);
  } else {
    return 'x';
  }
  return 'c';
}
