// RUN: %clangxx_msan -O0 %s -o %t && %run %t

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdlib.h>

void poison_stack_ahead() {
  char buf[100000];
  // With -O0 this poisons a large chunk of stack.
}

int main(void) {
  poison_stack_ahead();

  struct addrinfo *ai;

  // This should trigger loading of libnss_dns and friends.
  // Those libraries are typically uninstrumented.They will call strlen() on a
  // stack-allocated buffer, which is very likely to be poisoned. Test that we
  // don't report this as an UMR.
  int res = getaddrinfo("not-in-etc-hosts", NULL, NULL, &ai);
  return 0;
}
