// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=core,unix.StdCLibraryFunctions,unix.Errno \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -verify %s
//
// expected-no-diagnostics

#include "Inputs/errno_var.h"

typedef __typeof__(sizeof(int)) size_t;
char *getcwd(char *buf, size_t size);

void gh_175136(char *CharData, unsigned char *UCharData, size_t Size) {
  if (!getcwd(CharData, Size)) {
    if (errno == 2) {
      return;
    }
  }

  if (!getcwd((char*)UCharData, Size)) {
    if (errno == 2) {// Previously there was a (false) warning from unix.Errno on this line
      return;
    }
  }
}
