// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics

typedef union   {
  union wait *__uptr;
  int *__iptr;
} __WAIT_STATUS __attribute__ ((__transparent_union__));

extern int wait (__WAIT_STATUS __stat_loc);

void fastcgi_cleanup() {
  int status = 0;
  wait(&status);
}

