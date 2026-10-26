// RUN: split-file %s %t
// RUN: %clang -fno-sanitize=all -fPIC -shared %t/shared.c -o %t/shared.so
// RUN: %clangxx %t/main.cpp -o %t/main
// RUN: env LD_PRELOAD=%t/shared.so %run %t/main

// On Android, %t/main is the host android_run.py wrapper script, so
// `env LD_PRELOAD=%t/shared.so` preloads the target DSO into the host process
// (and android_run.py does not forward LD_PRELOAD to the device).
// UNSUPPORTED: android

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
