// RUN: %clangxx_tsan %s -fsanitize=thread -o %t
// RUN: %run %t 128
// RUN: not %run %t 129

#include <mutex>
#include <string>

int main(int argc, char *argv[]) {
  int num_of_mtx = std::stoi(argv[1]);

  std::mutex* mutexes = new std::mutex[num_of_mtx];

  for (int i = 0; i < num_of_mtx; i++) {
    mutexes[i].lock();
  }
  for (int i = 0; i < num_of_mtx; i++) {
    mutexes[i].unlock();
  }

  delete[] mutexes;
  return 0;
}
