// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t

#include <utmp.h>
#include <utmpx.h>
#include <sanitizer/msan_interface.h>

int main(void) {
  setutent();
  while (struct utmp *ut = getutent())
    __msan_check_mem_is_initialized(ut, sizeof(*ut));
  endutent();

  setutxent();
  while (struct utmpx *utx = getutxent())
    __msan_check_mem_is_initialized(utx, sizeof(*utx));
  endutxent();
}
