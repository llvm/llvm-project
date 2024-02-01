/* RUN: %clangxx_msan -O0 %s -o %t && %run %t 2>&1

   REQUIRES: target={{.*(linux|freebsd).*}}
*/

#include <assert.h>
#include <sys/eventfd.h>

#include <sanitizer/msan_interface.h>

int main(int argc, char *argv[]) {
  int efd = eventfd(42, 0);
  assert(efd >= 0);

  eventfd_t v;
  int ret = eventfd_read(efd, &v);
  assert(ret == 0);
  __msan_check_mem_is_initialized(&v, sizeof(v));

  assert(v == 42);
}
