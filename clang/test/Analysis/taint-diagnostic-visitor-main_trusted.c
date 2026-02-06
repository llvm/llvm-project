// RUN: %clang_analyze_cc1 -analyzer-checker=optin.taint,core,security.ArrayBound -analyzer-config assume-controlled-environment=true -analyzer-output=text -verify %s

// This file is for testing enhanced diagnostics produced by the GenericTaintChecker

typedef __typeof(sizeof(int)) size_t;
struct _IO_FILE;
typedef struct _IO_FILE FILE;

int scanf(const char *restrict format, ...);
int system(const char *command);
char* getenv( const char* env_var );
size_t strlen( const char* str );
char *strcat( char *dest, const char *src );
char * strncat ( char * destination, const char * source, size_t num );
char* strcpy( char* dest, const char* src );
char * strncpy ( char * destination, const char * source, size_t num );
void *malloc(size_t size );
void free( void *ptr );
char *fgets(char *str, int n, FILE *stream);
extern FILE *stdin;

// This is to test that in trusted env
// the diagnostics are constructed so
// that argc or argv are not marked as
// taint origin.
int main(int argc, char * argv[]) {
   if (argc < 1)// expected-note {{'argc' is >= 1}}
                // expected-note@-1 {{Taking false branch}}
     return 1;
   char cmd[2048] = "/bin/cat ";
   char filename[1024];
   scanf("%s", filename);// expected-note {{Taint originated here}}
                         // expected-note@-1 {{Taint propagated to the 2nd argument}}
   strncat(filename, argv[1], sizeof(filename)- - strlen(argv[1]) - 1);// expected-note {{Taint propagated to the 1st argument}}
   strncat(cmd, filename, sizeof(cmd) - strlen(cmd)-1);// expected-note {{Taint propagated to the 1st argument}}
   system(cmd);// expected-warning {{Untrusted data is passed to a system call}}
               // expected-note@-1 {{Untrusted data is passed to a system call}}
   return 0;
 }
