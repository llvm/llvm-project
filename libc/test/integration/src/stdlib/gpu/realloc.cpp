#include "test/IntegrationTest/test.h"

#include "src/__support/GPU/utils.h"
#include "src/stdlib/free.h"
#include "src/stdlib/malloc.h"
#include "src/stdlib/realloc.h"

using namespace LIBC_NAMESPACE;

TEST_MAIN(int, char **, char **) {
  // realloc(nullptr, size) is equivalent to malloc.
  int *alloc = reinterpret_cast<int *>(LIBC_NAMESPACE::realloc(nullptr, 32));
  EXPECT_NE(alloc, nullptr);
  *alloc = 42;
  EXPECT_EQ(*alloc, 42);

  // realloc to same size returns the same pointer.
  void *same = LIBC_NAMESPACE::realloc(alloc, 32);
  EXPECT_EQ(same, alloc);
  EXPECT_EQ(reinterpret_cast<int *>(same)[0], 42);

  // realloc to smaller size returns same pointer.
  void *smaller = LIBC_NAMESPACE::realloc(same, 16);
  EXPECT_EQ(smaller, alloc);
  EXPECT_EQ(reinterpret_cast<int *>(smaller)[0], 42);

  // realloc to larger size returns new pointer and preserves contents.
  int *larger = reinterpret_cast<int *>(LIBC_NAMESPACE::realloc(smaller, 128));
  EXPECT_NE(larger, nullptr);
  EXPECT_EQ(larger[0], 42);

  // realloc works when called with a divergent size.
  int *div = reinterpret_cast<int *>(
      LIBC_NAMESPACE::malloc((gpu::get_thread_id() + 1) * 16));
  EXPECT_NE(div, nullptr);
  div[0] = static_cast<int>(gpu::get_thread_id());
  int *div_realloc = reinterpret_cast<int *>(
      LIBC_NAMESPACE::realloc(div, ((gpu::get_thread_id() + 1) * 32)));
  EXPECT_NE(div_realloc, nullptr);
  EXPECT_EQ(div_realloc[0], static_cast<int>(gpu::get_thread_id()));
  LIBC_NAMESPACE::free(div_realloc);

  return 0;
}
