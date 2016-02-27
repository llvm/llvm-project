// RUN: %clangxx -frtti -fsanitize=vptr -fno-sanitize-recover=vptr -I%p/Helpers %p/Helpers/vptr-non-unique-typeinfo-lib.cpp -fPIC -shared -o %t-lib.so
// RUN: %clangxx -frtti -fsanitize=vptr -fno-sanitize-recover=vptr -I%p/Helpers -g %s -O3 -o %t %t-lib.so
// RUN: %run %t

#include "vptr-non-unique-typeinfo-lib.h"

int main() {
  X *px = libCall();
  delete px;
}
