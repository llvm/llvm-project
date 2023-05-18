/* This is a test case where the parent process forks 10
 * children which contend to write to the same file. With
 * file locking support, the data from each child should not
 * be lost.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

extern FILE *lprofOpenFileEx(const char *);
int main(int argc, char *argv[]) {
  pid_t tid;
  FILE *F;
  const char *FN;
  int child[10];
  int c;
  int i;

  if (argc < 2) {
    fprintf(stderr, "Requires one argument \n");
    exit(1);
  }
  FN = argv[1];
  truncate(FN, 0);

  for (i = 0; i < 10; i++) {
    c = fork();
    // in child: 
    if (c == 0) {
      FILE *F = lprofOpenFileEx(FN);
      if (!F) {
        fprintf(stderr, "Can not open file %s from child\n", FN);
        exit(1);
      }
      fseek(F, 0, SEEK_END);
      fprintf(F, "Dump from Child %d\n", i);
      fclose(F);
      exit(0);
    } else {
      child[i] = c;
    }
  }

  // In parent
  for (i = 0; i < 10; i++) {
    int child_status;
    if ((tid = waitpid(child[i], &child_status, 0)) == -1)
      break;
  }
  F = lprofOpenFileEx(FN);
  if (!F) {
    fprintf(stderr, "Can not open file %s from parent\n", FN);
    exit(1);
  }
  fseek(F, 0, SEEK_END);
  fprintf(F, "Dump from parent %d\n", i);
  return 0;
}
