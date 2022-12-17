#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main() { printf("main\n"); }

typedef unsigned long uptr;

#define FN(X)                                                                  \
  if (pc == reinterpret_cast<uptr>(X))                                         \
  return #X

const char *symbolize(uptr pc) {
  FUNCTIONS;
  return nullptr;
}

template <typename T> T consume(const char *&pos, const char *end) {
  T v = *reinterpret_cast<const T *>(pos);
  pos += sizeof(T);
  assert(pos <= end);
  return v;
}

uint32_t meta_version;
const char *meta_start;
const char *meta_end;

extern "C" {
void __sanitizer_metadata_covered_add(uint32_t version, const char *start,
                                      const char *end) {
  printf("metadata add version %u\n", version);
  for (const char *pos = start; pos < end;) {
    const uptr base = reinterpret_cast<uptr>(pos);
    const long offset = (version & (1 << 16)) ? consume<long>(pos, end)
                                              : consume<int>(pos, end);
    const uint32_t size = consume<uint32_t>(pos, end);
    const uint32_t features = consume<uint32_t>(pos, end);
    uint32_t stack_args = 0;
    if (features & (1 << 1))
      stack_args = consume<uint32_t>(pos, end);
    if (const char *name = symbolize(base + offset))
      printf("%s: features=%x stack_args=%u\n", name, features, stack_args);
  }
  meta_version = version;
  meta_start = start;
  meta_end = end;
}

void __sanitizer_metadata_covered_del(uint32_t version, const char *start,
                                      const char *end) {
  assert(version == meta_version);
  assert(start == meta_start);
  assert(end == meta_end);
}

const char *atomics_start;
const char *atomics_end;

void __sanitizer_metadata_atomics_add(uint32_t version, const char *start,
                                      const char *end) {
  assert(version == meta_version);
  assert(start);
  assert(end >= end);
  atomics_start = start;
  atomics_end = end;
}

void __sanitizer_metadata_atomics_del(uint32_t version, const char *start,
                                      const char *end) {
  assert(version == meta_version);
  assert(atomics_start == start);
  assert(atomics_end == end);
}
}
