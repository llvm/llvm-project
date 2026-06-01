// RUN: %clang_analyze_cc1 -analyzer-checker=optin.taint,core,security.ArrayBound -analyzer-config \
// RUN:   assume-controlled-environment=false -analyzer-output=text -verify %s

typedef __typeof(sizeof(int)) size_t;
int system(const char *command);
size_t strlen(const char *str);
char *strncat(char *destination, const char *source, size_t num);

// When main is declared with incorrect signature,
// the analyzer won't crash and we get a compilation error.

int main(char* argv[], int argc) {
  //expected-error@-1 {{first parameter of 'main' (argument count) must be of type 'int'}}
  //expected-error@-2 {{second parameter of 'main' (argument array) must be of type 'char **'}}
  char cmd[2048] = "/bin/cat ";
  char filename[1024];
  strncat(cmd, filename, sizeof(cmd) - strlen(cmd) - 1);
  system(cmd);
  return 0;
}
