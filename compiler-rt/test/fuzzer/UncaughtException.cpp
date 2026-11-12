#include <cstdint>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data, size_t size) {
  std::vector<uint8_t> v;
  // Intentionally throw std::length_error
  v.reserve(static_cast<uint64_t>(-1));

  return 0;
}
