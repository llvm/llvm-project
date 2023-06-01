// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-checker=alpha.unix.StdCLibraryFunctions \
// RUN:   -analyzer-checker=apiModeling.Errno \
// RUN:   -analyzer-checker=alpha.unix.Errno \
// RUN:   -analyzer-config alpha.unix.StdCLibraryFunctions:ModelPOSIX=true

#include "Inputs/errno_var.h"

typedef typeof(sizeof(int)) size_t;
typedef __typeof(sizeof(int)) off_t;
typedef size_t ssize_t;
ssize_t send(int sockfd, const void *buf, size_t len, int flags);
off_t lseek(int fildes, off_t offset, int whence);

void clang_analyzer_warnIfReached();
void clang_analyzer_eval(int);

int unsafe_errno_read(int sock, void *data, int data_size) {
  if (send(sock, data, data_size, 0) != data_size) {
    if (errno == 1) {
      // expected-warning@-1{{An undefined value may be read from 'errno'}}
      return 0;
    }
  }
  return 1;
}

int errno_lseek(int fildes, off_t offset) {
  off_t result = lseek(fildes, offset, 0);
  if (result == (off_t)-1) {
    // Failure path.
    // check if the function is modeled
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
    return 2;
  }
  if (result != offset) {
    // Not success path (?)
    // not sure if this is a valid case, allow to check 'errno'
    if (errno == 1) { // no warning
      return 1;
    }
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
  if (result == offset) {
    // The checker does not differentiate for this case.
    // In general case no relation exists between the arg 2 and the returned
    // value, only for SEEK_SET.
    if (errno == 1) { // no warning
      return 1;
    }
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
  return 0;
}
