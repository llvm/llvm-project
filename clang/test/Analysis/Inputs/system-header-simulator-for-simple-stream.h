// Like the compiler, the static analyzer treats some functions differently if
// they come from a system header -- for example, it is assumed that system
// functions do not arbitrarily free() their parameters, and that some bugs
// found in system headers cannot be fixed by the user and should be
// suppressed.
#pragma clang system_header

typedef __typeof(sizeof(int)) size_t;
typedef struct _FILE {
  unsigned char *_p;
} FILE;
FILE *fopen(const char *restrict, const char *restrict) __asm("_" "fopen" );
int fputc(int, FILE *);
int fputs(const char *restrict, FILE *restrict) __asm("_" "fputs" );
size_t fread(void *buffer, size_t size, size_t count, FILE *stream);
int fgetc(FILE *stream);
int fclose(FILE *);
void exit(int);

// The following is a fake system header function
typedef struct __FileStruct {
  FILE * p;
} FileStruct;
void fakeSystemHeaderCall(FileStruct *);
