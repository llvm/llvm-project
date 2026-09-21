// RUN: split-file %s %t
// RUN: %clang -fno-sanitize=all -fPIC -shared %t/shared.c -o %t/shared.so
// RUN: %clangxx %t/main.cpp -o %t/main
// RUN: env LD_PRELOAD=%t/shared.so %run %t/main

//--- shared.c
#include <dirent.h>

DIR *opendir(const char *path) {
  (void)path;
  return 0;
}

//--- main.cpp
#include <dirent.h>

int main() {
  const char *path = nullptr;
  return opendir(path) != nullptr;
}
