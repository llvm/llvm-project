#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main() { printf("main\n"); }

namespace {
#define FN(X)                                                                  \
  if (pc == reinterpret_cast<uintptr_t>(X))                                    \
  return #X

const char *symbolize(uintptr_t pc) {
  FUNCTIONS;
  return nullptr;
}

template <typename T> T consume(const char *&pos, const char *end) {
  T v;
  // We need to memcpy from pos, because it's not guaranteed that every entry
  // has the required alignment of T.
  memcpy(&v, pos, sizeof(T));
  pos += sizeof(T);
  assert(pos <= end);
  return v;
}

constexpr uint32_t kSanitizerBinaryMetadataUARHasSize = 1 << 2;

uint32_t meta_version;
const char *meta_start;
const char *meta_end;
const char *atomics_start;
const char *atomics_end;
}; // namespace

extern "C" {
void __sanitizer_metadata_covered_add(uint32_t version, const char *start,
                                      const char *end) {
  const uint32_t version_base = version & 0xffff;
  const bool offset_ptr_sized = version & (1 << 16);
  assert(version_base == 2);
  printf("metadata add version %u\n", version_base);
  for (const char *pos = start; pos < end;) {
    const auto base = reinterpret_cast<uintptr_t>(pos);
    const intptr_t offset = offset_ptr_sized ? consume<intptr_t>(pos, end)
                                             : consume<int32_t>(pos, end);
    [[maybe_unused]] const uint32_t size = consume<uint32_t>(pos, end);
    const uint32_t features = consume<uint8_t>(pos, end);
    uint32_t stack_args = 0;
    if (features & kSanitizerBinaryMetadataUARHasSize)
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
