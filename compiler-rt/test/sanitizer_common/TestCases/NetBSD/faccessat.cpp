// RUN: %clangxx -O0 -g %s -o %t && %run %t

#include <fcntl.h>
#include <unistd.h>

int main(void) { return faccessat(AT_FDCWD, "/root", F_OK, 0); }
