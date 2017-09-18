// RUN: %clangxx -O0 -g %s -lutil -o %t && %run %t | FileCheck %s

// REQUIRES: stable-runtime
// XFAIL: android && i386-target-arch && asan

#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#if __linux__
#include <pty.h>
#else
#include <util.h>
#endif

int
main (int argc, char** argv)
{
    int master;
    int pid = forkpty(&master, NULL, NULL, NULL);

    if(pid == -1) {
      fprintf(stderr, "forkpty failed\n");
      return 1;
    } else if (pid > 0) {
      char buf[1024];
      int res = read(master, buf, sizeof(buf));
      write(1, buf, res);
      write(master, "password\n", 9);
      while ((res = read(master, buf, sizeof(buf))) > 0) write(1, buf, res);
    } else {
      char *s = getpass("prompt");
      assert(strcmp(s, "password") == 0);
      write(1, "done\n", 5);
    }
    return 0;
}

// CHECK: prompt
// CHECK: done
