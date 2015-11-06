// RUN: %clang_cc1 -analyze -analyzer-checker=core,security.insecureAPI.vfork,unix.Vfork -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,security.insecureAPI.vfork,unix.Vfork -verify -x c++ %s

#include "Inputs/system-header-simulator.h"

void foo();

// Ensure that child process is properly checked.
int f1(int x) {
  pid_t pid = vfork(); // expected-warning{{Call to function 'vfork' is insecure}}
  if (pid != 0)
    return 0;

  switch (x) {
  case 0:
    // Ensure that modifying pid is ok.
    pid = 1; // no-warning
    // Ensure that calling whitelisted routines is ok.
    execl("", "", 0); // no-warning
    _exit(1); // no-warning
    break;
  case 1:
    // Ensure that writing variables is prohibited.
    x = 0; // expected-warning{{This assignment is prohibited after a successful vfork}}
    break;
  case 2:
    // Ensure that calling functions is prohibited.
    foo(); // expected-warning{{This function call is prohibited after a successful vfork}}
    break;
  default:
    // Ensure that returning from function is prohibited.
    return 0; // expected-warning{{Return is prohibited after a successful vfork; call _exit() instead}}
  }

  while(1);
}

// Same as previous but without explicit pid variable.
int f2(int x) {
  pid_t pid = vfork(); // expected-warning{{Call to function 'vfork' is insecure}}

  switch (x) {
  case 0:
    // Ensure that writing pid is ok.
    pid = 1; // no-warning
    // Ensure that calling whitelisted routines is ok.
    execl("", "", 0); // no-warning
    _exit(1); // no-warning
    break;
  case 1:
    // Ensure that writing variables is prohibited.
    x = 0; // expected-warning{{This assignment is prohibited after a successful vfork}}
    break;
  case 2:
    // Ensure that calling functions is prohibited.
    foo(); // expected-warning{{This function call is prohibited after a successful vfork}}
    break;
  default:
    // Ensure that returning from function is prohibited.
    return 0; // expected-warning{{Return is prohibited after a successful vfork; call _exit() instead}}
  }

  while(1);
}

// Ensure that parent process isn't restricted.
int f3(int x) {
  if (vfork() == 0) // expected-warning{{Call to function 'vfork' is insecure}}
    _exit(1);
  x = 0; // no-warning
  foo(); // no-warning
  return 0;
} // no-warning

// Unbound pids are special so test them separately.
void f4(int x) {
  switch (x) {
  case 0:
    vfork(); // expected-warning{{Call to function 'vfork' is insecure}}
    x = 0; // expected-warning{{This assignment is prohibited after a successful vfork}}
    break;

  case 1:
    {
      char args[2];
      switch (vfork()) { // expected-warning{{Call to function 'vfork' is insecure}}
      case 0:
        args[0] = 0; // expected-warning{{This assignment is prohibited after a successful vfork}}
        exit(1);
      }
      break;
    }

  case 2:
    {
      pid_t pid;
      if ((pid = vfork()) == 0) // expected-warning{{Call to function 'vfork' is insecure}}
        while(1); // no-warning
      break;
    }
  }
  while(1);
} //no-warning


void f5() {
  // See "libxtables: move some code to avoid cautions in vfork man page"
  // (http://lists.netfilter.org/pipermail/netfilter-buglog/2014-October/003280.html).
  if (vfork() == 0) { // expected-warning{{Call to function 'vfork' is insecure}}
    execl("prog", "arg1", 0); // no-warning
    exit(1);  // expected-warning{{This function call is prohibited after a successful vfork}}
  }
}

