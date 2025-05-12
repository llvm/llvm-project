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
  int x = 1;

  v.push_back(engine::Async([&x]{ x = 2;}));
}

