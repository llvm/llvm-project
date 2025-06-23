// RUN: %check_clang_tidy %s bugprone-taxi-async-use-after-free %t

namespace std {

template<typename T>
class vector {
public:
  void push_back(T) {}
};

}

namespace engine {

template <typename Function, typename... Args>
int Async(Function&& f, Args&&... args) {
  return 1;
}

}

void f_ok() {
  int x = 1;
  std::vector<int> v;

  v.push_back(engine::Async([&x]{ x = 2;}));
}

void f_use_after_free() {
  std::vector<int> v;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: std::vector<Task> can die after variable [bugprone-taxi-async-use-after-free]
  int x = 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable can be used after free [bugprone-taxi-async-use-after-free]

  v.push_back(engine::Async([&x]{ x = 2;}));
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: captured here [bugprone-taxi-async-use-after-free]
}

void f_ref() {
  int xx = 1;
  std::vector<int> v;
  int &x = x;

  v.push_back(engine::Async([&x]{ x = 2;}));
}

void f_ref_ref() {
  int xx = 1;
  std::vector<int> v;
  int &&x = static_cast<int&&>(xx);

  v.push_back(engine::Async([&x]{ x = 2;}));
}
