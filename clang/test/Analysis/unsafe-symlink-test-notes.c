// RUN: %clang_analyze_cc1 %s -triple=x86_64-unknown-linux \
// RUN:   -analyzer-output=text -verify \
// RUN:   -analyzer-checker=core,security.UnsafeSymlinkTest

struct stat {
  int st_mode;
  int st_ino;
  int st_dev;
};

typedef int size_t;
typedef size_t ssize_t;
int lstat(const char *restrict path, struct stat *restrict buf);
int open(const char *path, int oflag);
ssize_t write(int fildes, const void *buf, size_t nbyte);
int fstat(int fildes, struct stat *buf);

#define S_ISLNK(M) ((M & 2) != 0)

void test_fstat_single(const char *filename, const char *buf, size_t size) {
  struct stat stat1;
  int fd;

  if (lstat(filename, &stat1) == -1) // expected-note{{File status is read here before opening the file}} \\
                                     // expected-note{{Assuming the condition is false}} \\
                                     // expected-note{{Taking false branch}}
    return;

  fd = open(filename, 1);
  if (fd == -1) // expected-note{{Assuming the condition is false}} \\
                // expected-note{{Taking false branch}}
    return;

  struct stat stat2;
  if (fstat(fd, &stat2) == -1) // expected-note{{File status is read here after opening the file}} \\
                               // expected-note{{Assuming the condition is false}} \\
                               // expected-note{{Taking false branch}}
    return;

  write(fd, buf, size); // expected-warning{{Possibly missing check for external change of file}} \\
                        // expected-note{{Possibly missing check for external change of file}} \\
                        // expected-note{{File status was obtained before and after opening the file which indicates possible intent of a safe check for symbolic link}} \\
                        // expected-note{{For a safe check the fields 'st_mode', 'st_ino' and 'st_dev' before and after open should be checked for equality}}
}

void test_fstat_2(const char *fn2, const char *buf, size_t size) {
  const char *const fn1 = "x/y.z";
  struct stat lstat1;
  struct stat lstat2;
  int fd1, fd2;

  if (lstat(fn1, &lstat1) == -1) // expected-note{{File status of file 'x/y.z' is read here before opening the file}} \\
                                 // expected-note{{Assuming the condition is false}} \\
                                 // expected-note{{Taking false branch}}
    return;
  if (lstat(fn2, &lstat2) == -1) // expected-note{{Assuming the condition is false}} \\
                                 // expected-note{{Taking false branch}}
    return;

  fd1 = open(fn1, 1);
  if (fd1 == -1) // expected-note{{Assuming the condition is false}} \\
                 // expected-note{{Taking false branch}}
    return;
  fd2 = open(fn2, 1);
  if (fd2 == -1) // expected-note{{Assuming the condition is false}} \\
                 // expected-note{{Taking false branch}}
    return;

  struct stat fstat1;
  struct stat fstat2;
  if (fstat(fd1, &fstat1) == -1) // expected-note{{File status of file 'x/y.z' is read here after opening the file}} \\
                                 // expected-note{{Assuming the condition is false}} \\
                                 // expected-note{{Taking false branch}}
    return;
  if (fstat(fd2, &fstat2) == -1) // expected-note{{Assuming the condition is false}} \\
                                 // expected-note{{Taking false branch}}
    return;

  if (fstat2.st_mode == lstat2.st_mode && fstat2.st_ino == lstat2.st_ino && fstat2.st_dev == lstat2.st_dev) { // \\
  // expected-note{{Assuming 'fstat2.st_mode' is equal to 'lstat2.st_mode'}} \\
  // expected-note{{Left side of '&&' is true}} \\
  // expected-note{{Assuming 'fstat2.st_ino' is equal to 'lstat2.st_ino'}} \\
  // expected-note{{Left side of '&&' is true}} \\
  // expected-note{{Assuming 'fstat2.st_dev' is equal to 'lstat2.st_dev'}} \\
  // expected-note{{'fstat2.st_dev' is equal to 'lstat2.st_dev'}} \\
  // expected-note{{Taking true branch}}
    write(fd2, buf, size);
    write(fd1, buf, size); // expected-warning{{Possibly missing check for external change of file 'x/y.z'}} \\
                           // expected-note{{Possibly missing check for external change of file 'x/y.z'}} \\
                           // expected-note{{File status was obtained before and after opening the file which indicates possible intent of a safe check for symbolic link}} \\
                           // expected-note{{For a safe check the fields 'st_mode', 'st_ino' and 'st_dev' before and after open should be checked for equality}}
  }
}

const char *const g_filename = "a/b/c";

void test_islnk() {
  struct stat lstat_info;
  int fd;

  if (lstat(g_filename, &lstat_info) == -1) // expected-note{{File status of file 'a/b/c' is read here before opening the file}} \\
                                            // expected-note{{Assuming the condition is false}} \\
                                            // expected-note{{Taking false branch}}
    return;

  if (!S_ISLNK(lstat_info.st_mode)) { // expected-note{{Possible test if file 'a/b/c' is a symbolic link detected here}} \\
                                      // expected-note{{Assuming the condition is false}} \\
                                      // expected-note{{Taking true branch}}
    fd = open(g_filename, 1); // expected-warning{{Inaccurate check for symbolic link status of file 'a/b/c'}} \\
                              // expected-note{{Inaccurate check for symbolic link status of file 'a/b/c'}} \\
                              // expected-note{{The file can be manipulated externally between calling 'lstat' and opening the file}}
    if (fd == -1)
      return;
  }
}
