// Check that closedir(NULL) is ok.
// RUN: %clang -O2 %s -o %t && %run %t
#include <sys/types.h>
#include <dirent.h>
int main() { closedir(0); }
