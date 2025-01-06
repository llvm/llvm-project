#include "test_binary_properties.h"
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc != 1)
    return 5;
  constexpr size_t bufsize = sizeof(TEXT);
  char buf[bufsize];
  ssize_t readsize = bufsize - 1;
  ssize_t len = read(CHILD_FD, buf, readsize);
  if (len != readsize) {
    return 1;
  }
  buf[readsize] = '\0'; // Null terminator
  if (close(CHILD_FD) != 0)
    return 2;
  if (strcmp(buf, TEXT) != 0)
    return 3;
  return 0;
}
