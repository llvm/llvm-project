// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s < %t.out

#include <netdb.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>

int main(void) {
  const int nm = 4;
  struct gaicb *list[nm];

  for (auto i = 0; i < nm; i++) {
    list[i] = (struct gaicb *)malloc(sizeof(struct gaicb));
    if (i % 2)
      memset(list[i], 0, sizeof(struct gaicb));
    list[i]->ar_name = "name";
  }

  int res = getaddrinfo_a(GAI_WAIT, list, nm, NULL);
  for (auto i = 0; i < nm; i++) {
    // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
    // CHECK: {{#0 .* in main.*getaddrinfo_a.cpp:25}}
    for (auto it = list[i]->ar_result; it; it = it->ai_next) {
      auto name = it->ai_canonname;
    }

    free(list[i]);
  }
  return 0;
}
