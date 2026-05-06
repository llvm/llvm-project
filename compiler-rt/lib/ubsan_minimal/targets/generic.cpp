
#include "ubsan_minimal_common.h"

#include <stdlib.h>
#include <string.h>

#if defined(__ANDROID__)
extern "C" __attribute__((weak)) void android_set_abort_message(const char *);
#endif // defined(__ANDROID__)

#ifdef KERNEL_USE
extern "C" void ubsan_message(const char *msg);
static void message(const char *msg) { ubsan_message(msg); }
#elif defined(SANITIZER_AMDGPU) || defined(SANITIZER_NVPTX)
#include <stdio.h>
static void message(const char *msg) { fprintf(stderr, "%s", msg); }
#else
#include <unistd.h>
static void message(const char *msg) { (void)write(2, msg, strlen(msg)); }
#endif

static char *append_str(const char *s, char *buf, const char *end) {
  for (const char *p = s; (buf < end) && (*p != '\0'); ++p, ++buf)
    *buf = *p;
  return buf;
}

static char *append_hex(uintptr_t d, char *buf, const char *end) {
  // Print the address by nibbles.
  for (unsigned shift = sizeof(uintptr_t) * 8; shift && buf < end;) {
    shift -= 4;
    unsigned nibble = (d >> shift) & 0xf;
    *(buf++) = nibble < 10 ? nibble + '0' : nibble - 10 + 'a';
  }
  return buf;
}

static void format_msg(const char *kind, uintptr_t caller, char *buf,
                       const char *end) {
  buf = append_str("ubsan: ", buf, end);
  buf = append_str(kind, buf, end);
  buf = append_str(" by 0x", buf, end);
  buf = append_hex(caller, buf, end);
  buf = append_str("\n", buf, end);
  if (buf == end)
    --buf; // Make sure we don't cause a buffer overflow.
  *buf = '\0';
}

void __ubsan_message(const char *msg) {
  message(msg);
}

void __ubsan_message(const char *kind, uintptr_t caller) {
  char buf[128];
  format_msg(kind, caller, buf, buf + sizeof(buf));
  message(buf);
}

void __ubsan_abort() {
  abort();
}

void __ubsan_abort_with_message(const char *kind, uintptr_t caller) {
  char buf[128];
  format_msg(kind, caller, buf, buf + sizeof(buf));

#if defined(__ANDROID__)
  if (&android_set_abort_message)
    android_set_abort_message(buf);
#else // defined(__ANDROID__)
  message(buf);
#endif // defined(__ANDROID__)

  abort();
}
