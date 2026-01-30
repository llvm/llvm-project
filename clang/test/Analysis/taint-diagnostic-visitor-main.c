// RUN: %clang_analyze_cc1 -analyzer-checker=optin.taint,core,security.ArrayBound -DUNTRUSTED -analyzer-config assume-controlled-environment=false -analyzer-output=text -verify=expected,untrusted %s
// RUN: %clang_analyze_cc1 -analyzer-checker=optin.taint,core,security.ArrayBound -analyzer-config assume-controlled-environment=true -analyzer-output=text -verify=expected,trusted %s

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


#ifdef UNTRUSTED
// In an untrusted environment the cmd line arguments
// are assumed to be tainted.
int main(int argc, char * argv[]) {// untrusted-note {{Taint originated here}}
   if (argc < 1)// untrusted-note {{'argc' is >= 1}}
                // untrusted-note@-1 {{Taking false branch}}
     return 1;
   char cmd[2048] = "/bin/cat ";
   char filename[1024];
   strncpy(filename, argv[1], sizeof(filename)-1); // untrusted-note {{Taint propagated to the 1st argument}}
   strncat(cmd, filename, sizeof(cmd) - strlen(cmd)-1);// untrusted-note {{Taint propagated to the 1st argument}}
   system(cmd);// untrusted-warning {{Untrusted data is passed to a system call}}
               // untrusted-note@-1 {{Untrusted data is passed to a system call}}
   return 0;
 }
#else
int main(int argc, char * argv[]) {
   if (argc < 1)// trusted-note {{'argc' is >= 1}}
                // trusted-note@-1 {{Taking false branch}}
     return 1;
   char cmd[2048] = "/bin/cat ";
   char filename[1024];
   scanf("%s", filename);// trusted-note {{Taint originated here}}
                         // trusted-note@-1 {{Taint propagated to the 2nd argument}}
   strncat(filename, argv[1], sizeof(filename)- - strlen(argv[1]) - 1);// trusted-note {{Taint propagated to the 1st argument}}
   strncat(cmd, filename, sizeof(cmd) - strlen(cmd)-1);// trusted-note {{Taint propagated to the 1st argument}}
   system(cmd);// trusted-warning {{Untrusted data is passed to a system call}}
               // trusted-note@-1 {{Untrusted data is passed to a system call}}
   return 0;
 }
 #endif
