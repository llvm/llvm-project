// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t

#ifndef __FreeBSD__
#include <utmp.h>
#endif
#include <utmpx.h>
#include <sanitizer/msan_interface.h>

int main(void) {
#ifndef __FreeBSD__
  setutent();
  while (struct utmp *ut = getutent())
    __msan_check_mem_is_initialized(ut, sizeof(*ut));
  endutent();
#endif

  setutxent();
  while (struct utmpx *utx = getutxent())
    __msan_check_mem_is_initialized(utx, sizeof(*utx));
  endutxent();
}
