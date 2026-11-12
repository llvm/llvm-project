// RUN: %clangxx -O0 -g %s -o %t && %run %t

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) {
  int child;
  int fd, sfd;
  socklen_t len;
  struct sockaddr_in server = {}, client = {};
  sigset_t set;

  child = fork();
  if (child == 0) {
    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == -1)
      _exit(1);

    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(2222);

    if (connect(fd, (struct sockaddr *)&server, sizeof(server)) == -1)
      _exit(1);

    close(fd);

    _exit(0);
  }

  fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd == -1) {
    kill(child, SIGKILL);
    wait(NULL);
    exit(1);
  }

  server.sin_family = AF_INET;
  server.sin_addr.s_addr = INADDR_ANY;
  server.sin_port = htons(2222);

  if (bind(fd, (const struct sockaddr *)&server, sizeof(server)) == -1) {
    kill(child, SIGKILL);
    wait(NULL);
    exit(1);
  }

  listen(fd, 3);

  if (sigemptyset(&set) == -1) {
    kill(child, SIGKILL);
    wait(NULL);
    exit(1);
  }

  len = sizeof(client);
  sfd = paccept(fd, (struct sockaddr *)&client, &len, &set, SOCK_NONBLOCK);
  if (sfd == -1) {
    kill(child, SIGKILL);
    wait(NULL);
    exit(1);
  }

  wait(NULL);

  close(sfd);
  close(fd);

  return 0;
}
