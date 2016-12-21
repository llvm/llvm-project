// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.API -verify %s
extern "C" {
#ifndef O_RDONLY
#define O_RDONLY 0
#endif

#ifndef NULL
#define NULL ((void*) 0)
#endif

int open(const char *, int, ...);
int close(int fildes);

} // extern "C"

namespace MyNameSpace {
int open(const char *a, int b, int c, int d);
}

void unix_open(const char *path) {
  int fd;
  fd = open(path, O_RDONLY); // no-warning
  if (fd > -1)
    close(fd);
}

void unix_open_misuse(const char *path) {
  int fd;
  int mode = 0x0;
  fd = open(path, O_RDONLY, mode, NULL); // expected-warning{{Call to 'open' with more than 3 arguments}}
  if (fd > -1)
    close(fd);
}

// Don't treat open() in namespaces as the POSIX open()
void namespaced_open(const char *path) {
  MyNameSpace::open("Hi", 2, 3, 4); // no-warning

  using namespace MyNameSpace;

  open("Hi", 2, 3, 4); // no-warning

  int fd;
  int mode = 0x0;
  fd = ::open(path, O_RDONLY, mode, NULL); // expected-warning{{Call to 'open' with more than 3 arguments}}
  if (fd > -1)
    close(fd);
}

class MyClass {
public:
  static int open(const char *a, int b, int c, int d);

  int open(int a, int, int c, int d);
};

void class_qualified_open() {
  MyClass::open("Hi", 2, 3, 4); // no-warning

  MyClass mc;
  mc.open(1, 2, 3, 4); // no-warning
}
