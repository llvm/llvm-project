// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=core,unix.StdCLibraryFunctions,unix.Errno \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -verify %s
//
// expected-no-diagnostics

#include "Inputs/errno_var.h"

typedef __typeof__(sizeof(int)) size_t;
typedef struct {} FILE;
char *getcwd(char *buf, size_t size);
char *fgets(char *restrict s, int n, FILE *restrict stream);
char *mkdtemp(char *template);

int fputc(int c, FILE *stream);
#define EOF (-1)

void gh_175136(char *CharData, unsigned char *UCharData, size_t Size) {
  if (!getcwd(CharData, Size)) {
    if (errno == 2) {
      return;
    }
  }

  if (!getcwd((char*)UCharData, Size)) {
    if (errno == 2) { // Previously there was a (false) warning from unix.Errno on this line
      return;
    }
  }
}

void fgets_test(unsigned char *Buf, size_t Size, FILE *F) {
  if (!fgets((char *)Buf, Size, F))
    if (errno == 1) // Similar case for the (false) warning
      return;
}

void mkdtemp_test(unsigned char *Buf) {
  if (!mkdtemp((char *)Buf))
    if (errno == 1) // Similar case for the (false) warning
      return;
}

void fputc_test(long long c, FILE *stream) {
  if (fputc((int)c, stream) == EOF) {
    if (errno == 1) // This case did not produce the false warning
      return;
  }
}
