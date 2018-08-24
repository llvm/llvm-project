#include <thread>

int function(int x) {
  if ((x % 2) == 0)
    return function(x-1) + x; // breakpoint 1
  else
    return x;
}

int main(int argc, char const *argv[]) {
  std::thread thread1(function, 2);
  std::thread thread2(function, 4);
  thread1.join();
  thread2.join();
  return 0;
}
