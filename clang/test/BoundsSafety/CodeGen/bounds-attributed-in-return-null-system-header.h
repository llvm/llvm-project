// Pretend this is a system header
#pragma clang system_header
#include <ptrcheck.h>


inline int* __counted_by(count) inline_header_func_unspecified_ptr(int count) {
  // Outside of system headers this implicit conversion is not
  // allowed but it's allowed in system headers. This is a case
  // where an implicit cast to __bidi_indexable is needed.
  return (int*)0;
}

inline int* __counted_by(count) inline_header_func_unsafe_indexable_ptr(int count) {
  // Outside of system headers this implicit conversion is not
  // allowed but it's allowed in system headers. This is a case
  // where an implicit cast to __bidi_indexable is needed.
  return (int* __unsafe_indexable)0;
}

