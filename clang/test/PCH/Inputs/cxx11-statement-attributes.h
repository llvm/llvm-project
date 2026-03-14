// To be used with cxx11-statement-attributes.cpp.
template<const int N>
int f(int n) {
  switch (n * N) {
    case 0:
      n += 15;
      [[clang::fallthrough]];  // This shouldn't generate a warning.
    case 1:
      n += 20;
    case 2:  // This should generate a warning: "unannotated fallthrough"
      n += 35;
      break;
  }
  return n;
}
