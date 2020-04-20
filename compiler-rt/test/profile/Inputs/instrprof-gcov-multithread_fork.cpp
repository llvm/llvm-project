#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

template <typename T>
void launcher(T func) {
  std::vector<std::thread> pool;

  for (int i = 0; i < 10; i++) {
    pool.emplace_back(std::thread(func));
  }

  for (auto &t : pool) {
    t.join();
  }
}

void h() {}

void g() {
  fork();
  launcher<>(h);
}

void f() {
  fork();
  launcher<>(g);
}

int main() {
  launcher<>(f);

  return 0;
}
