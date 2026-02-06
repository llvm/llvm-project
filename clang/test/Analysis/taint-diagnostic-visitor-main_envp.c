// RUN: %clang_analyze_cc1 -analyzer-checker=optin.taint,core,security.ArrayBound -analyzer-config assume-controlled-environment=false -analyzer-output=text -verify %s

// This file is for testing enhanced diagnostics produced by the GenericTaintChecker

typedef __typeof(sizeof(int)) size_t;
struct _IO_FILE;
typedef struct _IO_FILE FILE;

int       atoi( const char* str );
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
#define PATH_MAX        4096

// In an untrusted environment the cmd line arguments
// are assumed to be tainted.
int main( int argc, char *argv[], char *envp[] ) {// expected-note {{Taint originated in 'argc'}}
   if (argc < 1)// expected-note {{'argc' is >= 1}}
                // expected-note@-1 {{Taking false branch}}
     return 1;         
   int v[5]={1,2,3,4,5};   
   return v[argc];// expected-warning {{Potential out of bound access to 'v' with tainted index}}
                  // expected-note@-1 {{Access of 'v' with a tainted index that may be too large}}
 }
