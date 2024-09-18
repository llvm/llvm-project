#include "clang/Basic/CodeGenOptions.h"
#include "llvm/Driver/Options.h"
#include "gtest/gtest.h"

TEST(VecLibBitfieldTest, AllLibrariesFit) {
  // We expect that all vector libraries fit in the bitfield size
  EXPECT_LE(static_cast<size_t>(llvm::driver::VectorLibrary::MaxLibrary),
            (1 << VECLIB_BIT_COUNT))
      << "VecLib bitfield size is too small!";
 }
