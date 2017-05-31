// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.API -verify %s

#ifndef O_RDONLY
#define O_RDONLY 0
#endif

#ifndef NULL
#define NULL ((void*) 0)
#endif

int open(const char *, int, ...);
int close(int fildes);

void open_1(const char *path) {
  int fd;
  fd = open(path, O_RDONLY); // no-warning
  if (fd > -1)
    close(fd);
}

void open_2(const char *path) {
  int fd;
  int mode = 0x0;
  fd = open(path, O_RDONLY, mode, NULL); // expected-warning{{Call to 'open' with more than three arguments}}
  if (fd > -1)
    close(fd);
}

void open_3(const char *path) {
  int fd;
  fd = open(path, O_RDONLY, NULL); // expected-warning{{Third argument to 'open' is not an integer}}
  if (fd > -1)
    close(fd);
}

void open_4(const char *path) {
  int fd;
  fd = open(path, O_RDONLY, ""); // expected-warning{{Third argument to 'open' is not an integer}}
  if (fd > -1)
    close(fd);
}

void open_5(const char *path) {
  int fd;
  struct {
    int val;
  } st = {0};
  fd = open(path, O_RDONLY, st); // expected-warning{{Third argument to 'open' is not an integer}}
  if (fd > -1)
    close(fd);
}

void open_6(const char *path) {
  int fd;
  struct {
    int val;
  } st = {0};
  fd = open(path, O_RDONLY, st.val); // no-warning
  if (fd > -1)
    close(fd);
}

void open_7(const char *path) {
  int fd;
  fd = open(path, O_RDONLY, &open); // expected-warning{{Third argument to 'open' is not an integer}}
  if (fd > -1)
    close(fd);
}

void open_8(const char *path) {
  int fd;
  fd = open(path, O_RDONLY, 0.0f); // expected-warning{{Third argument to 'open' is not an integer}}
  if (fd > -1)
    close(fd);
}
