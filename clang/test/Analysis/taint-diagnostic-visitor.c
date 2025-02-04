// RUN: %clang_cc1 -analyze -analyzer-checker=optin.taint,core,alpha.security.ArrayBoundV2 -analyzer-output=text -verify %s

// This file is for testing enhanced diagnostics produced by the GenericTaintChecker

typedef __typeof(sizeof(int)) size_t;
struct _IO_FILE;
typedef struct _IO_FILE FILE;

int scanf(const char *restrict format, ...);
int system(const char *command);
char* getenv( const char* env_var );
size_t strlen( const char* str );
char *strcat( char *dest, const char *src );
char* strcpy( char* dest, const char* src );
void *malloc(size_t size );
void free( void *ptr );
char *fgets(char *str, int n, FILE *stream);
extern FILE *stdin;

void taintDiagnostic(void)
{
  char buf[128];
  scanf("%s", buf); // expected-note {{Taint originated here}}
                    // expected-note@-1 {{Taint propagated to the 2nd argument}}
  system(buf); // expected-warning {{Untrusted data is passed to a system call}} // expected-note {{Untrusted data is passed to a system call (CERT/STR02-C. Sanitize data passed to complex subsystems)}}
}

int taintDiagnosticOutOfBound(void) {
  int index;
  int Array[] = {1, 2, 3, 4, 5};
  scanf("%d", &index); // expected-note {{Taint originated here}}
                       // expected-note@-1 {{Taint propagated to the 2nd argument}}
  return Array[index]; // expected-warning {{Potential out of bound access to 'Array' with tainted index}}
                       // expected-note@-1 {{Access of 'Array' with a tainted index}}
}

int taintDiagnosticDivZero(int operand) {
  scanf("%d", &operand); // expected-note {{Value assigned to 'operand'}}
                         // expected-note@-1 {{Taint originated here}}
                         // expected-note@-2 {{Taint propagated to the 2nd argument}}
  return 10 / operand; // expected-warning {{Division by a tainted value, possibly zero}}
                       // expected-note@-1 {{Division by a tainted value, possibly zero}}
}

void taintDiagnosticVLA(void) {
  int x;
  scanf("%d", &x); // expected-note {{Value assigned to 'x'}}
                   // expected-note@-1 {{Taint originated here}}
                   // expected-note@-2 {{Taint propagated to the 2nd argument}}
  int vla[x]; // expected-warning {{Declared variable-length array (VLA) has tainted}}
              // expected-note@-1 {{Declared variable-length array (VLA) has tainted}}
}


// Tests if the originated note is correctly placed even if the path is
// propagating through variables and expressions
int taintDiagnosticPropagation(){
  int res;
  char *cmd=getenv("CMD"); // expected-note {{Taint originated here}}
                           // expected-note@-1 {{Taint propagated to the return value}}
  if (cmd){ // expected-note {{Assuming 'cmd' is non-null}}
            // expected-note@-1 {{Taking true branch}}
    res = system(cmd); // expected-warning{{Untrusted data is passed to a system call}}
                       // expected-note@-1{{Untrusted data is passed to a system call}}
    return res;
  }
  return -1;
}

// Taint origin should be marked correctly even if there are multiple taint
// sources in the function
int taintDiagnosticPropagation2(){
  int res;
  char *user_env2=getenv("USER_ENV_VAR2");//unrelated taint source
  char *cmd=getenv("CMD"); // expected-note {{Taint originated here}}
                           // expected-note@-1 {{Taint propagated to the return value}}
  char *user_env=getenv("USER_ENV_VAR");//unrelated taint source
  if (cmd){ // expected-note {{Assuming 'cmd' is non-null}}
	          // expected-note@-1 {{Taking true branch}}
    res = system(cmd); // expected-warning{{Untrusted data is passed to a system call}}
                       // expected-note@-1{{Untrusted data is passed to a system call}}
    return res;
  }
  return 0;
}

void testReadStdIn(){
  char buf[1024];
  fgets(buf, sizeof(buf), stdin);// expected-note {{Taint originated here}}
                                 // expected-note@-1 {{Taint propagated to the 1st argument}}
  system(buf);// expected-warning {{Untrusted data is passed to a system call}}
              // expected-note@-1 {{Untrusted data is passed to a system call (CERT/STR02-C. Sanitize data passed to complex subsystems)}}

}

void multipleTaintSources(void) {
  char cmd[2048], file[1024];
  scanf ("%1022[^\n] ", cmd); // expected-note {{Taint originated here}}
                   // expected-note@-1 {{Taint propagated to the 2nd argument}}
  scanf ("%1023[^\n]", file); // expected-note {{Taint originated here}}
                   // expected-note@-1 {{Taint propagated to the 2nd argument}}
  strcat(cmd, file); // expected-note {{Taint propagated to the 1st argument}}
  strcat(cmd, " "); // expected-note {{Taint propagated to the 1st argument}}
  system(cmd); // expected-warning {{Untrusted data is passed to a system call}}
               // expected-note@-1{{Untrusted data is passed to a system call}}
}

void multipleTaintedArgs(void) {
  char cmd[1024], file[1024], buf[2048];
  scanf("%1022s %1023s", cmd, file); // expected-note {{Taint originated here}}
                          // expected-note@-1 {{Taint propagated to the 2nd argument, 3rd argument}}
  strcpy(buf, cmd);// expected-note {{Taint propagated to the 1st argument}}
  strcat(buf, " ");// expected-note {{Taint propagated to the 1st argument}}
  strcat(buf, file);// expected-note {{Taint propagated to the 1st argument}}
  system(buf); // expected-warning {{Untrusted data is passed to a system call}}
               // expected-note@-1{{Untrusted data is passed to a system call}}
}

void testTaintedMalloc(){
  size_t size = 0;
  scanf("%zu", &size); // expected-note {{Taint originated here}}
                       // expected-note@-1 {{Taint propagated to the 2nd argument}}
  int *p = malloc(size);// expected-warning{{malloc is called with a tainted (potentially attacker controlled) value}}
                    // expected-note@-1{{malloc is called with a tainted (potentially attacker controlled) value}}
  free(p);
}
