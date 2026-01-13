// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=core,unix.StdCLibraryFunctions,unix.Errno \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -verify %s
//
// expected-no-diagnostics

#include "Inputs/system-header-simulator-cxx.h"
#include "Inputs/errno_var.h"

char *getcwd(char *buf, size_t size);

int main(int argc, char *argv[]) {
  std::vector<char> charbuf;
  if (!getcwd(charbuf.data(), charbuf.size() - 1)) {
    if (errno == 2) {
      return 1;
    }
  }

  std::vector<unsigned char> ucharbuf;
  if (!getcwd((char*)ucharbuf.data(), ucharbuf.size() - 1)) {
    if (errno == 2) { // no (false) warning from unix.Errno on this line
      return 1;
    }
  }
  return 0;
}
