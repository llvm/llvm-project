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
ssize_t read(int fildes, void *buf, size_t nbyte);
int fstat(int fildes, struct stat *buf);

#define S_ISLNK(M) ((M & 2) != 0)

void test_simple(const char *filename, const char *buf, size_t size) {
  struct stat stat1;
  int fd;

  if (lstat(filename, &stat1) == -1) // expected-note{{File status is read here into 'stat1.st_mode' before opening the file}} \\
                                     // expected-note{{Assuming the condition is false}} \\
                                     // expected-note{{Taking false branch}}
    return;

  if (S_ISLNK(stat1.st_mode)) // expected-note{{Assuming the condition is false}} \\
                              // expected-note{{'stat1.st_mode' is checked here for symbolic link}} \\
                              // expected-note{{Taking false branch}}
    return;

  fd = open(filename, 1); // expected-note{{File is opened here}}
  if (fd == -1) // expected-note{{Assuming the condition is false}} \\
                // expected-note{{Taking false branch}}
    return;

  struct stat stat2;
  if (fstat(fd, &stat2) == -1) // expected-note{{File status is read here into 'stat2.st_mode' after opening the file}} \\
                               // expected-note{{Assuming the condition is false}} \\
                               // expected-note{{Taking false branch}}
    return;

  write(fd, buf, size); // expected-warning{{File might have been changed between call to 'lstat' and 'open' therefore 'stat1.st_mode' may not contain the state of the file at open}} \\
                        // expected-note{{File might have been changed between call to 'lstat' and 'open' therefore 'stat1.st_mode' may not contain the state of the file at open}}
}

void test_multiple(const char *fn2, const char *buf, size_t size) {
  const char *const fn1 = "x/y.z";
  struct stat lstat1;
  struct stat lstat2;
  int fd1, fd2;

  if (lstat(fn1, &lstat1) == -1) // expected-note{{File status of file 'x/y.z' is read here into 'lstat1.st_mode' before opening the file}} \\
                                 // expected-note{{Assuming the condition is false}} \\
                                 // expected-note{{Taking false branch}}
    return;
  if (lstat(fn2, &lstat2) == -1) // expected-note{{Assuming the condition is false}} \\
                                 // expected-note{{Taking false branch}}
    return;

  if (S_ISLNK(lstat1.st_mode) || S_ISLNK(lstat2.st_mode)) // expected-note{{Assuming the condition is false}} \\
                                                          // expected-note{{'lstat1.st_mode' is checked here for symbolic link}} \\
                                                          // expected-note{{Left side of '||' is false}} \\
                                                          // expected-note{{Assuming the condition is false}} \\
                                                          // expected-note{{Taking false branch}}
    return;

  fd1 = open(fn1, 1); // expected-note{{File 'x/y.z' is opened here}}
  if (fd1 == -1) // expected-note{{Assuming the condition is false}} \\
                 // expected-note{{Taking false branch}}
    return;
  fd2 = open(fn2, 1);
  if (fd2 == -1) // expected-note{{Assuming the condition is false}} \\
                 // expected-note{{Taking false branch}}
    return;

  struct stat fstat1;
  struct stat fstat2;
  if (fstat(fd1, &fstat1) == -1) // expected-note{{File status of file 'x/y.z' is read here into 'fstat1.st_mode' after opening the file}} \\
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
  // expected-note{{Taking true branch}}
    write(fd2, buf, size);
    write(fd1, buf, size); // expected-warning{{File 'x/y.z' might have been changed between call to 'lstat' and 'open' therefore 'lstat1.st_mode' may not contain the state of the file at open}} \\
                           // expected-note{{File 'x/y.z' might have been changed between call to 'lstat' and 'open' therefore 'lstat1.st_mode' may not contain the state of the file at open}}
  }
}

const char *const g_filename = "a/b/c";

void test_nofstat() {
  struct stat lstat_info;
  int fd;

  if (lstat(g_filename, &lstat_info) == -1) // expected-note{{File status of file 'a/b/c' is read here into 'lstat_info.st_mode' before opening the file}} \\
                                            // expected-note{{Assuming the condition is false}} \\
                                            // expected-note{{Taking false branch}}
    return;

  if (!S_ISLNK(lstat_info.st_mode)) { // expected-note{{'lstat_info.st_mode' is checked here for symbolic link}} \\
                                      // expected-note{{Assuming the condition is false}} \\
                                      // expected-note{{Taking true branch}}
    fd = open(g_filename, 1); // expected-note{{File 'a/b/c' is opened here}}
    if (fd == -1) // expected-note{{Assuming the condition is false}} \\
                  // expected-note{{Taking false branch}}
      return;

    char buf[10];
    read(fd, buf, 10); // expected-warning{{File 'a/b/c' might have been changed between call to 'lstat' and 'open' therefore 'lstat_info.st_mode' may not contain the state of the file at open}} \\
                       // expected-note{{File 'a/b/c' might have been changed between call to 'lstat' and 'open' therefore 'lstat_info.st_mode' may not contain the state of the file at open}}
  }
}

void test_stat_param(const char *filename, struct stat *lstatd) {
  int fd;

  if (lstat(filename, lstatd) == -1) // expected-note{{File status is read here into field 'st_mode' before opening the file}} \\
                                     // expected-note{{Assuming the condition is false}} \\
                                     // expected-note{{Taking false branch}}
    return;

  if (S_ISLNK(lstatd->st_mode)) // expected-note{{field 'st_mode' is checked here for symbolic link}} \\
                                // expected-note{{Assuming the condition is false}} \\
                                // expected-note{{Taking false branch}}
    return;

  fd = open(filename, 1); // expected-note{{File is opened here}}
  if (fd == -1) // expected-note{{Assuming the condition is false}} \\
                // expected-note{{Taking false branch}}
    return;

  char buf[10];
  read(fd, buf, 10); // expected-warning{{File might have been changed between call to 'lstat' and 'open' therefore field 'st_mode' may not contain the state of the file at open}} \\
                     // expected-note{{File might have been changed between call to 'lstat' and 'open' therefore field 'st_mode' may not contain the state of the file at open}}
}
