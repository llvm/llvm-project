// To be used with cxx11-statement-attributes.cpp.
template<const int N>
int f(int n) {
  switch (n * N) {
    case 0:
      n += 15;
      [[clang::fallthrough]];  // This shouldn't generate a warning.
    case 1:
      n += 20;
      [[clang::fallthrough]];  // This should generate a warning: "fallthrough annotation does not directly precede switch label".
      break;
  }
  return n;
}
