#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
  char *env = getenv("EXECV_TEST");
  if (env == nullptr)
    raise(SIGUSR1);
  return 0;
}
