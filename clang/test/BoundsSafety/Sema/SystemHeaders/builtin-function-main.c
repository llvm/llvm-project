
#include <ptrcheck.h>
#include <builtin-function-sys.h>

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -verify -I %S/include
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -verify -I %S/include -x objective-c -fexperimental-bounds-safety-objc
// expected-no-diagnostics

char * __counted_by(len) func(char * __counted_by(len) src_str, int len) {
  int len2 = 0;
  char * __counted_by(len2) dst_str;
  dst_str = __unsafe_forge_bidi_indexable(char*, malloc(len), len);
  len2 = len;
  memcpy(dst_str, src_str, len);
  return dst_str;
}
