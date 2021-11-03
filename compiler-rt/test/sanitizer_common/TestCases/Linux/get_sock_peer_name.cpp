// Test that ASan doesn't raise false alarm when getsockname and getpeername
// are called with addrlen=nullptr;
//
// RUN: %clangxx %s -o %t && %run %t 2>&1

// connect() fails on Android.
// UNSUPPORTED: android

#include <assert.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>

int main() {
  const int fd = socket(AF_INET, SOCK_DGRAM, 0);
  assert(fd >= 0);

  const sockaddr_in sin = {
      .sin_family = AF_INET,
      .sin_port = 1234,
      .sin_addr =
          {
              .s_addr = INADDR_LOOPBACK,
          },
  };
  assert(connect(fd, reinterpret_cast<const sockaddr *>(&sin), sizeof(sin)) ==
         0);

  errno = 0;
  assert(getsockname(fd, nullptr, nullptr) == -1);
  assert(errno == EFAULT);

  errno = 0;
  assert(getpeername(fd, nullptr, nullptr) == -1);
  assert(errno == EFAULT);

  return 0;
}
