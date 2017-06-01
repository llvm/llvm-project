// RUN: %clang_esan_frag -O0 %s -o %t 2>&1
// RUN: %env_esan_opts=verbosity=3 %run %t 2>&1 | FileCheck %s

#include <string.h>

int main(int argc, char **argv) {
  char Buf[2048];
  const char Str[] = "TestStringOfParticularLength"; // 29 chars.
  strcpy(Buf, Str);
  strncpy(Buf, Str, 17);
  return strncmp(Buf, Str, 17);
  // CHECK:      in esan::initializeLibrary
  // CHECK:      in esan::processRangeAccess {{.*}} 29
  // CHECK:      in esan::processRangeAccess {{.*}} 29
  // CHECK:      in esan::processRangeAccess {{.*}} 17
  // CHECK:      in esan::processRangeAccess {{.*}} 17
  // CHECK:      in esan::processRangeAccess {{.*}} 17
  // CHECK:      in esan::processRangeAccess {{.*}} 17
  // CHECK:      in esan::finalizeLibrary
}
