// RUN: %clang_analyze_cc1 -analyzer-checker=optin.taint,core,security.ArrayBound \
// RUN: -analyzer-config assume-controlled-environment=false -analyzer-output=text -verify %s

// This file is for testing enhanced diagnostics produced by the
// GenericTaintChecker

typedef __typeof(sizeof(int)) size_t;
int system(const char *command);
size_t strlen(const char *str);
char *strncat(char *destination, const char *source, size_t num);
char *strncpy(char *destination, const char *source, size_t num);

// In an untrusted environment the the environment variables
// coming through the envp are also tainted.
int main(int argc, char *argv[],  char *envp[]) { // expected-note {{Taint originated in 'envp'}}
  char cmd[2048] = "/bin/cat ";
  char filename[1024];
  if (!envp[0]) // expected-note {{Assuming the condition is false}}
                // expected-note@-1 {{Taking false branch}}
    return 1;
  strncpy(filename, envp[0], sizeof(filename) - 1); // expected-note {{Taint propagated to the 1st argument}}
  strncat(cmd, filename, sizeof(cmd) - strlen(cmd) - 1); // expected-note {{Taint propagated to the 1st argument}}
  system(cmd); // expected-warning {{Untrusted data is passed to a system call}}
               // expected-note@-1 {{Untrusted data is passed to a system call}}
  return 0;
}
