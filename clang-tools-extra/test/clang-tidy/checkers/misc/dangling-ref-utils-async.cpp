// RUN: %check_clang_tidy %s misc-dangling-ref-utils-async %t

namespace std {

template <typename T>
class vector {
 public:
  void push_back(T);
};

}  // namespace std

namespace utils {

class Task {};

template <typename T>
class TaskWithResult : public Task {};

template <class T>
TaskWithResult<int> Async(T) {}

}  // namespace utils

void f1() {
  std::vector<utils::Task> tasks;
  int x = 1;
  tasks.push_back(utils::Async([&x] {}));
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: Might be a use-after-delete if exception is thrown inside lambda [misc-dangling-ref-utils-async]
}

void f2() {
  std::vector<utils::Task> tasks;
  for (;;) {
    int x = 1;
    tasks.push_back(utils::Async([&x] {}));
    // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: Might be a use-after-delete if exception is thrown inside lambda [misc-dangling-ref-utils-async]
  }
}

// Not a match: x dies after tasks
void f3() {
  int x = 1;
  std::vector<utils::Task> tasks;
  tasks.push_back(utils::Async([&x] {}));
}

// Not a match: x is copied
void f4() {
  std::vector<utils::Task> tasks;
  int x = 1;
  tasks.push_back(utils::Async([x] {}));
}

// Not a match: x is copied
void f5() {
  std::vector<utils::Task> tasks;
  int x = 1;
  tasks.push_back(utils::Async([x] {}));
}

