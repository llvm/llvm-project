// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

class Logger {
 public:
  Logger() {
    fprintf(stderr, "Logger ctor\n");
  }

  ~Logger() {
    fprintf(stderr, "Logger dtor\n");
  }
};

Logger logger;

void log_from_atexit() {
  fprintf(stderr, "In log_from_atexit\n");
}

int main() {
  atexit(log_from_atexit);
}

// CHECK: Logger ctor
// CHECK: In log_from_atexit
// CHECK: Logger dtor
