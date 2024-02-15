#include <signal.h> // SIGUSR1
#include <stdlib.h> // getenv
#include <unistd.h> // raise

int main() {
  char *env = getenv("__MISSING_ENV_VAR__");
  if (env == nullptr)
    raise(SIGUSR1);
  return 0;
}
