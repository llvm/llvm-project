#include "../code_object_utils.hpp"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

namespace {

// Minimal "ELF-shaped garbage" that does not parse as a valid AMDGPU
// code object: the magic bytes are correct but everything else is zero.
std::vector<uint8_t> garbageElf() {
  return {0x7f, 'E', 'L', 'F', 0, 0, 0, 0};
}

} // namespace

TEST(CodeObjectUtils, EmptyDataParsesAsEmpty) {
  std::vector<uint8_t> empty;
  EXPECT_TRUE(transpiler::listKernelNames(empty).empty());
  EXPECT_FALSE(transpiler::extractTextSection(empty).valid);
  EXPECT_TRUE(transpiler::detectIsaFromElf(empty).empty());
}

TEST(CodeObjectUtils, MalformedElfYieldsNoKernels) {
  auto data = garbageElf();
  EXPECT_TRUE(transpiler::listKernelNames(data).empty());
  EXPECT_FALSE(transpiler::extractTextSection(data).valid);
  EXPECT_TRUE(transpiler::detectIsaFromElf(data).empty());
}

TEST(CodeObjectUtils, MissingKernelMetaIsDefaulted) {
  auto data = garbageElf();
  transpiler::KernelMeta meta =
      transpiler::extractKernelMeta(data, "missing_kernel");
  EXPECT_FALSE(meta.hasKernelDescriptor);
  EXPECT_EQ(meta.kernargSegmentSize, 0);
}
