#include <sys/types.h>
#include <thread>
#include <unistd.h>

template <typename T>
void launcher(T func) {
  auto t1 = std::thread(func);
  auto t2 = std::thread(func);

  t1.join();
  t2.join();
}

void g() {}

void f() {
  fork();
  launcher<>(g);
}

int main() {
  launcher<>(f);

  return 0;
}
