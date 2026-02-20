// RUN: %clang_analyze_cc1 -analyzer-checker=optin.taint,core,security.ArrayBound -analyzer-config \
// RUN: assume-controlled-environment=false -analyzer-output=text -verify %s

// This file is for testing enhanced 
// diagnostics produced by the GenericTaintChecker

typedef __typeof(sizeof(int)) size_t;
int system(const char *command);
size_t strlen(const char *str);
char *strncat(char *destination, const char *source, size_t num);

// invalid main function
// expected-no-diagnostics

int main(void) {
  char cmd[2048] = "/bin/cat ";
  char filename[1024];
  strncat(cmd, filename, sizeof(cmd) - strlen(cmd) - 1);
  system(cmd);
  return 0;
}
