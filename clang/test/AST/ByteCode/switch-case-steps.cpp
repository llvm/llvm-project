// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++23 %s -fconstexpr-steps=2000
// RUN: %clang_cc1                                         -std=c++23 %s -fconstexpr-steps=2000



constexpr int char_to_int(char ch) {
  switch (ch) {
  case '0':
    return 0;
  case '1':
    return 1;
  case '2':
    return 2;
  case '3':
    return 3;
  case '4':
    return 4;
  case '5':
    return 5;
  case '6':
    return 6;
  case '7':
    return 7;
  case '8':
    return 8;
  case '9':
    return 9;
  case 'a':
  case 'A':
    return 10;
  case 'b':
  case 'B':
    return 11;
  case 'c':
  case 'C':
    return 12;
  case 'd':
  case 'D':
    return 13;
  case 'e':
  case 'E':
    return 14;
  case 'f':
  case 'F':
    return 15;
  case 'g':
  case 'G':
    return 16;
  case 'h':
  case 'H':
    return 17;
  case 'i':
  case 'I':
    return 18;
  case 'j':
  case 'J':
    return 19;
  case 'k':
  case 'K':
    return 20;
  case 'l':
  case 'L':
    return 21;
  case 'm':
  case 'M':
    return 22;
  case 'n':
  case 'N':
    return 23;
  case 'o':
  case 'O':
    return 24;
  case 'p':
  case 'P':
    return 25;
  case 'q':
  case 'Q':
    return 26;
  case 'r':
  case 'R':
    return 27;
  case 's':
  case 'S':
    return 28;
  case 't':
  case 'T':
    return 29;
  case 'u':
  case 'U':
    return 30;
  case 'v':
  case 'V':
    return 31;
  case 'w':
  case 'W':
    return 32;
  case 'x':
  case 'X':
    return 33;
  case 'y':
  case 'Y':
    return 34;
  case 'z':
  case 'Z':
    return 35;
  default:
    return 0;
  }
}

constexpr bool check() {
  const char *str = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n";
  unsigned sum = 0;
  for (const char *p = str; *p != '\0'; ++p) {
    sum+= char_to_int(*p);
  }

  return sum != 0;
}
static_assert(check());
