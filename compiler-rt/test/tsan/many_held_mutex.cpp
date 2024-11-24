// RUN: %clangxx_tsan -O1 %s %link_libcxx_tsan -fsanitize=thread -o %t
// RUN: %run %t 128
// RUN: not %run %t 129

#include <mutex>
#include <vector>
#include <string>

int main(int argc, char *argv[]) {
  int num_of_mtx = std::atoi(argv[1]);

  std::vector<std::mutex> mutexes(num_of_mtx);

  for (auto& mu : mutexes) {
    mu.lock();
  }
  for (auto& mu : mutexes) {
    mu.unlock();
  }

  return 0;
}
