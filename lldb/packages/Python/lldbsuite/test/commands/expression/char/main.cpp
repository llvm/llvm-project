int foo(char c) { return 1; }
int foo(signed char c) { return 2; }
int foo(unsigned char c) { return 3; }

int main() {
  char c = 0;
  signed char sc = 0;
  unsigned char uc = 0;
  return 0; // Break here
}
