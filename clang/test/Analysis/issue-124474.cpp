// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=core,unix.BlockInCriticalSection \
// RUN:   -analyzer-output text -verify %s

// expected-no-diagnostics

#include "Inputs/system-header-simulator-cxx-std-locks.h"

std::mutex mtx;
using ssize_t = long long;
using size_t = unsigned long long;
int open(const char *__file, int __oflag, ...);
ssize_t read(int fd, void *buf, size_t count);
void close(int fd);
#define O_RDONLY 00
#define O_NONBLOCK 04000

void foo() {
  std::lock_guard<std::mutex> lock(mtx);

  const char *filename = "example.txt";
  int fd = open(filename, O_RDONLY | O_NONBLOCK);

  char buffer[200] = {};
  read(fd, buffer, 199); // no-warning: fd is a non-block file descriptor or equals to -1
  close(fd);
}

void foo1(int fd) {
  std::lock_guard<std::mutex> lock(mtx);

  const char *filename = "example.txt";
  char buffer[200] = {};
  if (fd == -1)
    read(fd, buffer, 199); // no-warning: consider file descriptor is a symbol equals to -1
  close(fd);
}
