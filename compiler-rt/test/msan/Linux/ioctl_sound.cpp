// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t
// RUN: %clangxx_msan -O3 -g %s -o %t && %run %t

#include <assert.h>
#include <fcntl.h>
#include <sound/asound.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <sanitizer/msan_interface.h>

int main(int argc, char **argv) {
  int fd = open("/dev/snd/controlC0", O_RDONLY);
  if (fd < 0) {
    printf("Unable to open sound device.");
    return 0;
  }
  const unsigned sz = sizeof(snd_ctl_card_info);
  void *info = malloc(sz + 1);
  assert(__msan_test_shadow(info, sz) == 0);
  assert(ioctl(fd, SNDRV_CTL_IOCTL_CARD_INFO, info) >= 0);
  assert(__msan_test_shadow(info, sz + 1) == sz);
  close(fd);
  free(info);
  return 0;
}
