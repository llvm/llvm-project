#include <stdatomic.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHILDREN 7

int main(int argc, char *argv[]) {
  _Atomic int *sync = mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (sync == MAP_FAILED)
    return 1;
  *sync = 0;

  for (int i = 0; i < CHILDREN; i++) {
    pid_t pid = fork();
    if (!pid) {
      // child
      while (*sync == 0)
        ; // wait the parent in order to call execl simultaneously
      execl(argv[1], argv[1], NULL);
    } else if (pid == -1) {
      *sync = 1; // release all children
      return 1;
    }
  }

  // parent
  *sync = 1; // start the program in all children simultaneously
  for (int i = 0; i < CHILDREN; i++)
    wait(NULL);

  return 0;
}
