// RUN: %clang_analyze_cc1 %s -triple=x86_64-unknown-linux \
// RUN:   -verify -analyzer-config eagerly-assume=false \
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

void test_lstat_islnk_open(const char *filename) {
  struct stat lstat_info;
  int fd;

  if (lstat(filename, &lstat_info) == -1)
    return;

  if (!S_ISLNK(lstat_info.st_mode)) {
    fd = open(filename, 1);
    if (fd == -1)
      return;

    char buf[10];
    read(fd, buf, 10); // expected-warning{{File might have been changed between call to 'lstat' and 'open' therefore 'lstat_info.st_mode' may not contain the state of the file at open}}
  }
}

void test_lstat_islnk_open_fstat(const char *filename) {
  struct stat lstat_info, fstat_info;
  int fd;

  if (lstat(filename, &lstat_info) == -1)
    return;

  if (!S_ISLNK(lstat_info.st_mode)) {
    fd = open(filename, 1);
    if (fd == -1)
      return;

    if (fstat(fd, &fstat_info) != -1) {
      char buf[10];
      read(fd, buf, 10); // expected-warning{{File might have been changed between call to 'lstat' and 'open' therefore 'lstat_info.st_mode' may not contain the state of the file at open}}
    }
  }
}

void test_lstat_open_islnk_fstat_nocompare() {
  struct stat lstat_info, fstat_info;
  int fd;

  if (lstat("filename", &lstat_info) == -1)
    return;

  fd = open("filename", 1);
  if (fd == -1)
    return;

  if (!S_ISLNK(lstat_info.st_mode) && fstat(fd, &fstat_info) != -1 && lstat_info.st_mode == fstat_info.st_mode) {
    char buf[10];
    read(fd, buf, 10); // expected-warning{{File 'filename' might have been changed between call to 'lstat' and 'open' therefore 'lstat_info.st_mode' may not contain the state of the file at open}}
  }
}

void test_lstat_open_islnk_fstat_nocompare_p(struct stat *fstat_info) {
  struct stat lstat_info;
  int fd;

  if (lstat("filename", &lstat_info) == -1)
    return;

  fd = open("filename", 1);
  if (fd == -1)
    return;

  if (!S_ISLNK(lstat_info.st_mode) && fstat(fd, fstat_info) != -1 && lstat_info.st_mode == fstat_info->st_mode) {
    char buf[10];
    read(fd, buf, 10); // expected-warning{{File 'filename' might have been changed between call to 'lstat' and 'open' therefore 'lstat_info.st_mode' may not contain the state of the file at open}}
  }
}

void test_lstat_open_fstat_noislnk(const char *filename) {
  struct stat lstat_info, fstat_info;
  int fd;

  if (lstat(filename, &lstat_info) == -1)
    return;

  fd = open(filename, 1);
  if (fd == -1)
    return;

  if (fstat(fd, &fstat_info) != -1 && !S_ISLNK(fstat_info.st_mode)) {
    char buf[10];
    read(fd, buf, 10); // no-warning
  }
}

void test_lstat_islnk_open_fstat_compare(const char *filename) {
  struct stat lstat_info, fstat_info;
  int fd;

  if (lstat(filename, &lstat_info) == -1)
    return;

  if (!S_ISLNK(lstat_info.st_mode)) {
    fd = open(filename, 1);
    if (fd == -1)
      return;

    if (fstat(fd, &fstat_info) != -1 && lstat_info.st_mode == fstat_info.st_mode && lstat_info.st_ino == fstat_info.st_ino && lstat_info.st_dev == fstat_info.st_dev) {
      char buf[10];
      read(fd, buf, 10); // no-warning
    }
  }
}

void test_lstat_islnk_open_fstat_compare_other(const char *filename) {
  struct stat lstat_info, fstat_info;
  int fd;

  if (lstat(filename, &lstat_info) == -1)
    return;

  if (S_ISLNK(lstat_info.st_mode))
    return;

  fd = open(filename, 1);
  if (fd == -1)
    return;

  if (fstat(fd, &fstat_info) == -1)
    return;

  if (lstat_info.st_mode != fstat_info.st_mode || lstat_info.st_dev != fstat_info.st_dev || fstat_info.st_ino != lstat_info.st_ino)
    return;

  char buf[10];
  read(fd, buf, 10); // no-warning
}

void test_lstat_islnk_open_fstat_compare_p(const char *filename, struct stat *fstat_info) {
  struct stat lstat_info;
  int fd;

  if (lstat(filename, &lstat_info) == -1)
    return;

  if (!S_ISLNK(lstat_info.st_mode)) {
    fd = open(filename, 1);
    if (fd == -1)
      return;

    if (fstat(fd, fstat_info) != -1 && fstat_info->st_mode == lstat_info.st_mode && lstat_info.st_ino == fstat_info->st_ino && fstat_info->st_dev == lstat_info.st_dev) {
      char buf[10];
      read(fd, buf, 10); // no-warning
    }
  }
}

void test_lstat_open_noislnk(const char *filename) {
  struct stat lstat_info, fstat_info;
  int fd;

  if (lstat(filename, &lstat_info) == -1)
    return;

  fd = open(filename, 1);
  if (fd == -1)
    return;

  char buf[10];
  read(fd, buf, 10); // no-warning
}

const char *const GlobalFName = "aaa/bbb";

void test_stat_param(struct stat *lstatd) {
  int fd;

  if (lstat(GlobalFName, lstatd) == -1)
    return;

  if (S_ISLNK(lstatd->st_mode))
    return;

  fd = open(GlobalFName, O_OTHER);
  if (fd == -1)
    return;

  char buf[10];
  read(fd, buf, 10); // expected-warning{{File 'aaa/bbb' might have been changed between call to 'lstat' and 'open' therefore field 'st_mode' may not contain the state of the file at open}}
}

void test_nofollow(const char *filename) {
  struct stat lstat_info;
  int fd;

  if (lstat(filename, &lstat_info) == -1)
    return;

  if (!S_ISLNK(lstat_info.st_mode)) {
    fd = open(filename, O_NOFOLLOW | O_OTHER);
    if (fd == -1)
      return;
    char buf[10];
    read(fd, buf, 10); // no-warning
  }
}

void test_nofollow_unknown(int flags) {
  struct stat lstat_info;
  int fd;

  if (lstat("f", &lstat_info) == -1)
    return;

  if (!S_ISLNK(lstat_info.st_mode)) {
    fd = open("f", flags);
    if (fd == -1)
      return;
    char buf[10];
    read(fd, buf, 10); // expected-warning{{File 'f' might have been changed between call to 'lstat' and 'open'}}
  }
}

void test_another_file(const char *filename, struct stat *lstat_info1, struct stat *lstat_info2) {
  int fd;

  if (lstat(filename, lstat_info1) == -1)
    return;

  if (lstat("file", lstat_info2) == -1)
    return;

  if (!S_ISLNK(lstat_info2->st_mode)) {
    fd = open(filename, 1);
    if (fd == -1)
      return;
    char buf[10];
    read(fd, buf, 10); // no-warning
  }
}

void test_more_files(const char *filename1, const char *filename2) {
  struct stat lstat_info1;
  struct stat lstat_info2;

  if (lstat(filename1, &lstat_info1) == -1)
    return;
  if (lstat(filename2, &lstat_info2) == -1)
    return;

  if (!S_ISLNK(lstat_info1.st_mode) && !S_ISLNK(lstat_info2.st_mode)) {
    int fd1 = open(filename1, 1);
    if (fd1 == -1)
      return;
    int fd2 = open(filename2, 1);
    if (fd2 == -1)
      return;

    char buf[10];
    read(fd1, buf, 10); // expected-warning{{File might have been changed between call to 'lstat' and 'open'}}
    read(fd2, buf, 10); // expected-warning{{File might have been changed between call to 'lstat' and 'open'}}
  }
}

extern void f_stat(struct stat *);
extern void f_fd(int *);

void test_lstat_inval_open(const char *buf, size_t size) {
  struct stat stat1;
  int fd;

  if (lstat("a/b", &stat1) == -1)
    return;

  if (S_ISLNK(stat1.st_mode))
    return;

  f_stat(&stat1);

  fd = open("a/b", 1);
  if (fd == -1)
    return;

  write(fd, buf, size); // no-warning
}

void test_lstat_open_inval(const char *buf, size_t size) {
  struct stat stat1;
  int fd;

  if (lstat("a/b", &stat1) == -1)
    return;

  if (S_ISLNK(stat1.st_mode))
    return;

  fd = open("a/b", 1);
  if (fd == -1)
    return;

  f_stat(&stat1);

  write(fd, buf, size); // no-warning
}

void test_lstat_open_fstat_inval(const char *buf, size_t size) {
  struct stat stat1, stat2;
  int fd;

  if (lstat("a/b", &stat1) == -1)
    return;

  if (S_ISLNK(stat1.st_mode))
    return;

  fd = open("a/b", 1);
  if (fd == -1)
    return;

  if (fstat(fd, &stat2) == -1)
    return;

  f_stat(&stat2);

  write(fd, buf, size); // no-warning
}

void test_inval_fd(const char *buf, size_t size) {
  struct stat stat1;
  int fd;

  if (lstat("a/b", &stat1) == -1)
    return;

  if (S_ISLNK(stat1.st_mode))
    return;

  fd = open("a/b", 1);
  if (fd == -1)
    return;

  f_fd(&fd);

  write(fd, buf, size); // no-warning
}

void test_inval_p(struct stat *stat1, const char *buf, size_t size) {
  int fd;

  if (lstat("a/b", stat1) == -1)
    return;

  if (S_ISLNK(stat1->st_mode))
    return;

  f_stat(stat1);

  fd = open("a/b", 1);
  if (fd == -1)
    return;

  write(fd, buf, size); // no-warning
}
