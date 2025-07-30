#include "test/IntegrationTest/test.h"

#include "src/__support/GPU/utils.h"
#include "src/stdlib/aligned_alloc.h" // Adjust path if needed
#include "src/stdlib/free.h"

using namespace LIBC_NAMESPACE;

TEST_MAIN(int, char **, char **) {
  // aligned_alloc with valid alignment and size
  void *ptr = LIBC_NAMESPACE::aligned_alloc(32, 16);
  EXPECT_NE(ptr, nullptr);
  EXPECT_TRUE(__builtin_is_aligned(ptr, 32));

  LIBC_NAMESPACE::free(ptr);

  // aligned_alloc fails if alignment is not power of two
  void *bad_align = LIBC_NAMESPACE::aligned_alloc(30, 99);
  EXPECT_EQ(bad_align, nullptr);

  // aligned_alloc with a divergent size.
  size_t alignment = 1 << (__gpu_lane_id() % 8 + 1);
  void *div =
      LIBC_NAMESPACE::aligned_alloc(alignment, (gpu::get_thread_id() + 1) * 4);
  EXPECT_NE(div, nullptr);
  EXPECT_TRUE(__builtin_is_aligned(div, alignment));

  return 0;
}
