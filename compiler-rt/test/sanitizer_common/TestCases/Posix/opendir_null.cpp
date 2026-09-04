// REQUIRES: linux

// RUN: split-file %s %t
// RUN: %clangxx -fno-sanitize=all -fPIC -shared %t/shared.cpp -o %t/shared.so
// RUN: %clangxx %t/main.cpp -o %t/main
// RUN: env LD_PRELOAD=%t/shared.so %run %t/main

//--- shared.cpp
#include <dirent.h>

extern "C" DIR *opendir(const char *) { return nullptr; }

//--- main.cpp
#include <dirent.h>

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : nullptr;
  return opendir(path) != nullptr;
}
