int main() {
  int *p = 0;
  *p = 7; // We expect a diagnostic about this.
  return 0;
}
