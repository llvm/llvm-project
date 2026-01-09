#include <coroutine>
#include <functional>
#include <iostream>
#include <queue>
#include <thread>

std::queue<std::function<bool()>> task_queue;

struct co_sleep {
  co_sleep(int n) : delay{n} {}

  constexpr bool await_ready() const noexcept { return false; }

  void await_suspend(std::coroutine_handle<> h) const noexcept {
    auto start = std::chrono::steady_clock::now();
    task_queue.push([start, h, d = delay] {
      if (decltype(start)::clock::now() - start > d) {
        h.resume();
        return true;
      } else {
        return false;
      }
    });
  }

  void await_resume() const noexcept {}

  std::chrono::milliseconds delay;
};

struct Task {
  struct promise_type {
    promise_type() = default;
    Task get_return_object() { return {}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void unhandled_exception() {}
  };
};

Task foo1() noexcept {
  std::cout << "1. hello from foo1" << std::endl;
  for (int i = 0; i < 10; ++i) {
    co_await co_sleep{10};
    std::cout << "2. hello from foo1" << std::endl;
  }
}

Task foo2() noexcept {
  std::cout << "1. hello from foo2" << std::endl;
  for (int i = 0; i < 10; ++i) {
    co_await co_sleep{10};
    std::cout << "2. hello from foo2" << std::endl;
  }
}

// call foo
int main() {
  foo1();
  foo2();

  while (!task_queue.empty()) {
    auto task = task_queue.front();
    if (!task()) {
      task_queue.push(task);
    }
    task_queue.pop();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
