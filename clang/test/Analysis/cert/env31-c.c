// RUN: %clang_analyze_cc1 -analyzer-output=text -Wno-unused %s     \
// RUN:   -analyzer-checker=core,alpha.security.cert.env.InvalidPtr \
// RUN:   -verify=putenv,common                                     \
// RUN:   -DENV_INVALIDATING_CALL="putenv(\"X=Y\")"
//
// RUN: %clang_analyze_cc1 -analyzer-output=text -Wno-unused %s     \
// RUN:   -analyzer-checker=core,alpha.security.cert.env.InvalidPtr \
// RUN:   -verify=putenvs,common                                    \
// RUN:   -DENV_INVALIDATING_CALL="_putenv_s(\"X\", \"Y\")"
//
// RUN: %clang_analyze_cc1 -analyzer-output=text -Wno-unused %s     \
// RUN:   -analyzer-checker=core,alpha.security.cert.env.InvalidPtr \
// RUN:   -verify=wputenvs,common                                   \
// RUN:   -DENV_INVALIDATING_CALL="_wputenv_s(\"X\", \"Y\")"
//
// RUN: %clang_analyze_cc1 -analyzer-output=text -Wno-unused %s     \
// RUN:   -analyzer-checker=core,alpha.security.cert.env.InvalidPtr \
// RUN:   -verify=setenv,common                                     \
// RUN:   -DENV_INVALIDATING_CALL="setenv(\"X\", \"Y\", 0)"
//
// RUN: %clang_analyze_cc1 -analyzer-output=text -Wno-unused %s     \
// RUN:   -analyzer-checker=core,alpha.security.cert.env.InvalidPtr \
// RUN:   -verify=unsetenv,common                                   \
// RUN:   -DENV_INVALIDATING_CALL="unsetenv(\"X\")"

typedef int errno_t;
typedef char wchar_t;

int putenv(char *string);
errno_t _putenv_s(const char *varname, const char *value_string);
errno_t _wputenv_s(const wchar_t *varname, const wchar_t *value_string);
int setenv(const char *name, const char *value, int overwrite);
int unsetenv(const char *name);

void fn_without_body(char **e);
void fn_with_body(char **e) {}

void call_env_invalidating_fn(char **e) {
  ENV_INVALIDATING_CALL;
  // putenv-note@-1 5 {{'putenv' call may invalidate the environment parameter of 'main'}}
  // putenvs-note@-2 5 {{'_putenv_s' call may invalidate the environment parameter of 'main'}}
  // wputenvs-note@-3 5 {{'_wputenv_s' call may invalidate the environment parameter of 'main'}}
  // setenv-note@-4 5 {{'setenv' call may invalidate the environment parameter of 'main'}}
  // unsetenv-note@-5 5 {{'unsetenv' call may invalidate the environment parameter of 'main'}}

  *e;
  // common-warning@-1 {{dereferencing an invalid pointer}}
  // common-note@-2 {{dereferencing an invalid pointer}}
}

int main(int argc, char *argv[], char *envp[]) {
  char **e = envp;
  *e;    // no-warning
  e[0];  // no-warning
  *envp; // no-warning
  call_env_invalidating_fn(e);
  // common-note@-1 5 {{Calling 'call_env_invalidating_fn'}}
  // common-note@-2 4 {{Returning from 'call_env_invalidating_fn'}}

  *e;
  // common-warning@-1 {{dereferencing an invalid pointer}}
  // common-note@-2 {{dereferencing an invalid pointer}}

  *envp;
  // common-warning@-1 2 {{dereferencing an invalid pointer}}
  // common-note@-2 2 {{dereferencing an invalid pointer}}

  fn_without_body(e);
  // common-warning@-1 {{use of invalidated pointer 'e' in a function call}}
  // common-note@-2 {{use of invalidated pointer 'e' in a function call}}

  fn_with_body(e); // no-warning
}
