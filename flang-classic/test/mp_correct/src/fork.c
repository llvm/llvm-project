#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

void fork_(int *pid)
{
  *pid = fork();
}

void waitpid_(int *pid)
{
  int status;

  waitpid(*pid, &status, 0);
  if (WCOREDUMP(status)) {
    fprintf(stderr, "FAIL\n");
    exit(-1);
  }
}
