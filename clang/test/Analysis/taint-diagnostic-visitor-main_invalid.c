// RUN: %clang_analyze_cc1 -analyzer-checker=optin.taint,core,security.ArrayBound -analyzer-config assume-controlled-environment=false -analyzer-output=text -verify %s
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


// invalid main function
// expected-no-diagnostics

int main(void) {   
   char cmd[2048] = "/bin/cat ";
   char filename[1024];   
   strncat(cmd, filename, sizeof(cmd) - strlen(cmd)-1);
   system(cmd);               
   return 0;
}
