#include "src/sys/random/getrandom.h"

// TODO: define this
inline int getentropy(void *buf, size_t size) {
  while (size > 0) {
    ssize_t ret = LIBC_NAMESPACE::getrandom(buf, size, 0);
    if (ret < 0) {
      return -1;
    }
    buf = (char *)buf + ret;
    size -= ret;
  }
  return 0;
}
