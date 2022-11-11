#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
  char *env = getenv("__MISSING_ENV_VAR__");
  if (env == nullptr)
    raise(SIGUSR1);
  return 0;
}
