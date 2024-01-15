// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-checker=apiModeling.Errno \
// RUN:   -analyzer-checker=unix.Errno \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true

#include "Inputs/errno_var.h"
#include "Inputs/std-c-library-functions-POSIX.h"

#define NULL ((void *) 0)

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

void errno_mkstemp(char *template) {
  int FD = mkstemp(template);
  if (FD >= 0) {
    if (errno) {}                    // expected-warning{{An undefined value may be read from 'errno'}}
    close(FD);
  } else {
    clang_analyzer_eval(FD == -1);   // expected-warning{{TRUE}}
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
    if (errno) {}                    // no warning
  }
}

void errno_mkdtemp(char *template) {
  char *Dir = mkdtemp(template);
  if (Dir == NULL) {
    clang_analyzer_eval(errno != 0);      // expected-warning{{TRUE}}
    if (errno) {}                         // no warning
  } else {
    clang_analyzer_eval(Dir == template); // expected-warning{{TRUE}}
    if (errno) {}                         // expected-warning{{An undefined value may be read from 'errno'}}
  }
}

void errno_getcwd(char *Buf, size_t Sz) {
  char *Path = getcwd(Buf, Sz);
  if (Sz == 0) {
    clang_analyzer_eval(errno != 0);   // expected-warning{{TRUE}}
    clang_analyzer_eval(Path == NULL); // expected-warning{{TRUE}}
    if (errno) {}                      // no warning
  } else if (Path == NULL) {
    clang_analyzer_eval(errno != 0);   // expected-warning{{TRUE}}
    if (errno) {}                      // no warning
  } else {
    clang_analyzer_eval(Path == Buf);  // expected-warning{{TRUE}}
    if (errno) {}                      // expected-warning{{An undefined value may be read from 'errno'}}
  }
}
