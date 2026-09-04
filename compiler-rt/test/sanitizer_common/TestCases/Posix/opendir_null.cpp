// REQUIRES: linux

// RUN: %clangxx -fno-sanitize=all -DSHARED_LIB -fPIC -shared %s -o %t.so
// RUN: %clangxx %s -o %t
// RUN: env LD_PRELOAD=%t.so %run %t

#include <dirent.h>

#if defined(SHARED_LIB)
extern "C" DIR *opendir(const char *) { return nullptr; }
#else
int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : nullptr;
  return opendir(path) != nullptr;
}
#endif
