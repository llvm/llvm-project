// RUN: %clang_analyze_cc1 %s -triple=x86_64-unknown-linux \
// RUN:   -verify \
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
#define O_NOFOLLOW (4)
#define O_OTHER (2)

void test_islnk_local(const char *filename) {
  struct stat lstat_info;
  int fd;

  if (lstat(filename, &lstat_info) == -1)
    return;

  if (!S_ISLNK(lstat_info.st_mode)) {
    fd = open(filename, 1); // expected-warning{{Inaccurate check for symbolic link status of file}} \\
                            // expected-note{{The file can be manipulated externally between calling 'lstat' and opening the file}}
    if (fd == -1)
      return;
  }
}

void test_islnk_param(const char *filename, struct stat *lstat_info) {
  int fd;

  if (lstat(filename, lstat_info) == -1)
    return;

  if (!S_ISLNK(lstat_info->st_mode)) {
    fd = open(filename, O_OTHER); // expected-warning{{Inaccurate check for symbolic link status of file}} \\
                                  // expected-note{{The file can be manipulated externally between calling 'lstat' and opening the file}}
    if (fd == -1)
      return;
  }
}

void test_no_islnk(const char *filename) {
  struct stat lstat_info;
  int fd;

  if (lstat(filename, &lstat_info) == -1)
    return;

  if (lstat_info.st_mode > 1) {
    fd = open(filename, O_OTHER); // no-warning
    if (fd == -1)
      return;
  }
}

void test_islnk_nofollow(const char *filename) {
  struct stat lstat_info;
  int fd;

  if (lstat(filename, &lstat_info) == -1)
    return;

  if (!S_ISLNK(lstat_info.st_mode)) {
    fd = open(filename, O_NOFOLLOW | O_OTHER); // no-warning
    if (fd == -1)
      return;
  }
}

void test_lstat_other(const char *filename, struct stat *lstat_info1, struct stat *lstat_info2) {
  int fd;

  if (lstat(filename, lstat_info1) == -1)
    return;

  if (lstat("file", lstat_info2) == -1)
    return;

  if (!S_ISLNK(lstat_info2->st_mode)) {
    fd = open(filename, 1); // no-warning
    if (fd == -1)
      return;
  }
}

void test_lstat_multi(const char *filename1, const char *filename2) {
  struct stat lstat_info1;
  struct stat lstat_info2;

  if (lstat(filename1, &lstat_info1) == -1)
    return;
  if (lstat(filename2, &lstat_info2) == -1)
    return;

  if (!S_ISLNK(lstat_info1.st_mode) && !S_ISLNK(lstat_info2.st_mode)) {
    int fd1 = open(filename1, 1); // expected-warning{{Inaccurate check for symbolic link status of file}} \\
                                  // expected-note{{The file can be manipulated externally between calling 'lstat' and opening the file}}
    if (fd1 == -1)
      return;
    int fd2 = open(filename2, 1); // expected-warning{{Inaccurate check for symbolic link status of file}} \\
                                  // expected-note{{The file can be manipulated externally between calling 'lstat' and opening the file}}
    if (fd2 == -1)
      return;
  }
}

void test_lstat_str_const() {
  struct stat lstat_info;
  int fd;

  if (lstat("x/y", &lstat_info) == -1)
    return;

  if (!S_ISLNK(lstat_info.st_mode)) {
    fd = open("x/y", 1); // expected-warning{{Inaccurate check for symbolic link status of file}} \\
                         // expected-note{{The file can be manipulated externally between calling 'lstat' and opening the file}}
    if (fd == -1)
      return;
  }
}

void test_fstat_nocheck(const char *filename, char *buf, size_t size) {
  struct stat lstat_info;
  int fd;

  if (lstat(filename, &lstat_info) == -1)
    return;

  fd = open(filename, 1);
  if (fd == -1)
    return;

  struct stat stat1;
  if (fstat(fd, &stat1) == -1)
    return;

  read(fd, buf, size); // expected-warning{{Possibly missing check for external change of file}} \\
                       // expected-note{{File status was obtained before and after opening the file which indicates possible intent of a safe check for symbolic link}} \\
                       // expected-note{{For a safe check the fields 'st_mode', 'st_ino' and 'st_dev' before and after open should be checked for equality}}
}

void test_fstat_badcheck(const char *filename, const char *buf, size_t size) {
  struct stat stat1;
  int fd;

  if (lstat(filename, &stat1) == -1)
    return;

  fd = open(filename, 1);
  if (fd == -1)
    return;

  struct stat stat2;
  if (fstat(fd, &stat2) == -1)
    return;

  if (stat1.st_mode == stat2.st_mode)
    write(fd, buf, size); // expected-warning{{Possibly missing check for external change of file}} \\
                          // expected-note{{File status was obtained before and after opening the file which indicates possible intent of a safe check for symbolic link}} \\
                          // expected-note{{For a safe check the fields 'st_mode', 'st_ino' and 'st_dev' before and after open should be checked for equality}}
}

void test_fstat_badcheck_p(const char *filename, const char *buf, size_t size, struct stat *stat1, struct stat *stat2) {
  int fd;

  if (lstat(filename, stat1) == -1)
    return;

  fd = open(filename, 1);
  if (fd == -1)
    return;

  if (fstat(fd, stat2) == -1)
    return;

  if (stat1->st_mode == stat2->st_mode)
    write(fd, buf, size); // expected-warning{{Possibly missing check for external change of file}} \\
                          // expected-note{{File status was obtained before and after opening the file which indicates possible intent of a safe check for symbolic link}} \\
                          // expected-note{{For a safe check the fields 'st_mode', 'st_ino' and 'st_dev' before and after open should be checked for equality}}
}

void test_fstat_goodcheck(const char *filename, const char *buf, size_t size) {
  struct stat stat1;
  int fd;

  if (lstat(filename, &stat1) == -1)
    return;

  fd = open(filename, 1);
  if (fd == -1)
    return;

  struct stat stat2;
  if (fstat(fd, &stat2) == -1)
    return;

  if (stat1.st_mode == stat2.st_mode && stat1.st_ino == stat2.st_ino && stat1.st_dev == stat2.st_dev)
    write(fd, buf, size); // no-warning
}

void test_fstat_goodcheck_p(const char *filename, const char *buf, size_t size, struct stat *stat1, struct stat *stat2) {
  int fd;

  if (lstat(filename, stat1) == -1)
    return;

  fd = open(filename, 1);
  if (fd == -1)
    return;

  if (fstat(fd, stat2) == -1)
    return;

  if (stat1->st_mode == stat2->st_mode && stat1->st_ino == stat2->st_ino && stat1->st_dev == stat2->st_dev)
    write(fd, buf, size); // no-warning
}

void test_fstat_nofollow_p(const char *filename, const char *buf, size_t size, struct stat *stat1, struct stat *stat2) {
  int fd;

  if (lstat(filename, stat1) == -1)
    return;

  fd = open(filename, O_NOFOLLOW);
  if (fd == -1)
    return;

  if (fstat(fd, stat2) == -1)
    return;

  write(fd, buf, size); // no-warning
}

void test_fstat_nofollow_unknown(const char *filename, const char *buf, size_t size, int flags) {
  int fd;
  struct stat stat1;
  struct stat stat2;

  if (lstat(filename, &stat1) == -1)
    return;

  fd = open(filename, flags);
  if (fd == -1)
    return;

  if (fstat(fd, &stat2) == -1)
    return;

  write(fd, buf, size); // no-warning
}

extern void f_stat(struct stat *);
extern void f_fd(int *);

void test_fstat_inval1(const char *filename, const char *buf, size_t size) {
  struct stat stat_e1;
  int fd;

  if (lstat(filename, &stat_e1) == -1)
    return;

  f_stat(&stat_e1);

  fd = open(filename, 1);
  if (fd == -1)
    return;

  struct stat stat2;
  if (fstat(fd, &stat2) == -1)
    return;

  write(fd, buf, size);
}

void test_fstat_inval2(const char *filename, const char *buf, size_t size) {
  struct stat stat1;
  int fd;

  if (lstat(filename, &stat1) == -1)
    return;

  fd = open(filename, 1);
  if (fd == -1)
    return;

  f_stat(&stat1);

  struct stat stat2;
  if (fstat(fd, &stat2) == -1)
    return;

  write(fd, buf, size);
}

void test_fstat_inval3(const char *filename, const char *buf, size_t size) {
  struct stat stat1;
  int fd;

  if (lstat(filename, &stat1) == -1)
    return;

  fd = open(filename, 1);
  if (fd == -1)
    return;

  struct stat stat2;
  if (fstat(fd, &stat2) == -1)
    return;

  f_stat(&stat1);

  write(fd, buf, size);
}

void test_fstat_inval4(const char *filename, const char *buf, size_t size) {
  struct stat stat1;
  int fd;

  if (lstat(filename, &stat1) == -1)
    return;

  fd = open(filename, 1);
  if (fd == -1)
    return;

  struct stat stat2;
  if (fstat(fd, &stat2) == -1)
    return;

  f_stat(&stat2);

  write(fd, buf, size);
}

void test_fstat_inval5(const char *filename, const char *buf, size_t size) {
  struct stat stat1;
  int fd;

  if (lstat(filename, &stat1) == -1)
    return;

  fd = open(filename, 1);
  if (fd == -1)
    return;

  struct stat stat2;
  if (fstat(fd, &stat2) == -1)
    return;

  f_fd(&fd);

  write(fd, buf, size);
}

void test_fstat_inval_p(const char *filename, const char *buf, size_t size, struct stat *stat1, struct stat *stat2) {
  int fd;

  if (lstat(filename, stat1) == -1)
    return;

  fd = open(filename, 1);
  if (fd == -1)
    return;

  f_stat(stat1);

  if (fstat(fd, stat2) == -1)
    return;

  if (stat1->st_mode == stat2->st_mode && stat1->st_ino == stat2->st_ino && stat1->st_dev == stat2->st_dev)
    write(fd, buf, size); // no-warning
}

void test_islnk_inval1_p(const char *filename, struct stat *lstat_info) {
  int fd;

  if (lstat(filename, lstat_info) == -1)
    return;

  f_stat(lstat_info);

  if (!S_ISLNK(lstat_info->st_mode)) {
    fd = open(filename, 1); // no-warning
    if (fd == -1)
      return;
  }
}

void test_islnk_inval2_p(const char *filename, struct stat *lstat_info) {
  int fd;

  if (lstat(filename, lstat_info) == -1)
    return;

  if (!S_ISLNK(lstat_info->st_mode)) {
    f_stat(lstat_info);
    fd = open(filename, 1); // expected-warning{{Inaccurate check for symbolic link status of file}} \\
                            // expected-note{{The file can be manipulated externally between calling 'lstat' and opening the file}}
    if (fd == -1)
      return;
  }
}
