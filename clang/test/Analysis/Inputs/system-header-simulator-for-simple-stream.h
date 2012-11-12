
#pragma clang system_header

typedef struct __sFILE {
  unsigned char *_p;
} FILE;
FILE *fopen(const char * restrict, const char * restrict) __asm("_" "fopen" );
int fputc(int, FILE *);
int fputs(const char * restrict, FILE * restrict) __asm("_" "fputs" );
int fclose(FILE *);
void exit(int);
