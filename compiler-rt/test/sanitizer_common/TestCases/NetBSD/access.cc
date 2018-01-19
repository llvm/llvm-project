// RUN: %clangxx -O0 -g %s -o %t && %run %t

#include <unistd.h>

int main(void) { return access("/root", F_OK); }
