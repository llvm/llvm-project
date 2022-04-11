// RUN: %clangxx %s -o %t -lresolv && %run %t %p

// -lresolv fails on Android.
// UNSUPPORTED: android

#include <assert.h>
#include <resolv.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

int main(int iArgc, char *szArgv[]) {
  // Check NTOP writing
  const char *src = "base64 test data";
  size_t src_len = strlen(src);
  size_t dst_len = ((src_len + 2) / 3) * 4 + 1;
  char dst[dst_len];
  int res = b64_ntop(reinterpret_cast<const unsigned char *>(src), src_len, dst,
                     dst_len);
  assert(res >= 0);
  assert(strcmp(dst, "YmFzZTY0IHRlc3QgZGF0YQ==") == 0);

  // Check PTON writing
  unsigned char target[dst_len];
  res = b64_pton(dst, target, dst_len);
  assert(res >= 0);
  assert(strncmp(reinterpret_cast<const char *>(target), src, res) == 0);

  // Check NTOP writing for zero length src
  src = "";
  src_len = strlen(src);
  assert(((src_len + 2) / 3) * 4 + 1 < dst_len);
  res = b64_ntop(reinterpret_cast<const unsigned char *>(src), src_len, dst,
                 dst_len);
  assert(res >= 0);
  assert(strcmp(dst, "") == 0);

  // Check PTON writing for zero length src
  dst[0] = '\0';
  res = b64_pton(dst, target, dst_len);
  assert(res >= 0);
  assert(strncmp(reinterpret_cast<const char *>(target), src, res) == 0);

  return 0;
}
