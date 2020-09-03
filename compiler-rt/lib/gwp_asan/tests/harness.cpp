#include "harness.h"

namespace gwp_asan {
namespace test {
bool OnlyOnce() {
  static int x = 0;
  return !x++;
}
} // namespace test
} // namespace gwp_asan
