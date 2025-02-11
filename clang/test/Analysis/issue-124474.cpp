// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=unix.BlockInCriticalSection \
// RUN:   -std=c++11 \
// RUN:   -analyzer-output text \
// RUN:   -verify %s

// expected-no-diagnostics

namespace std {
  struct mutex {
    void lock() {}
    void unlock() {}
  };
  template<typename T>
  struct lock_guard {
    lock_guard<T>(std::mutex) {}
    ~lock_guard<T>() {}
  };
  template<typename T>
  struct unique_lock {
    unique_lock<T>(std::mutex) {}
    ~unique_lock<T>() {}
  };
  template<typename T>
  struct not_real_lock {
    not_real_lock<T>(std::mutex) {}
  };
  } // namespace std

std::mutex mtx;
using ssize_t = long long;
using size_t = unsigned long long;
int open (const char *__file, int __oflag, ...);
ssize_t read(int fd, void *buf, size_t count);
void close(int fd);
#define O_RDONLY	     00
# define O_NONBLOCK	  04000

void foo()
{
    std::lock_guard<std::mutex> lock(mtx);

    const char *filename = "example.txt";
    int fd = open(filename, O_RDONLY | O_NONBLOCK);

    char buffer[200] = {};
    read(fd, buffer, 199); // no-warning: fd is a non-block file descriptor
    close(fd);
}