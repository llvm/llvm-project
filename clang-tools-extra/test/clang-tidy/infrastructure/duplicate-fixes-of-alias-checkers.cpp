// RUN: %check_clang_tidy %s cppcoreguidelines-pro-type-member-init,hicpp-member-init,modernize-use-emplace,hicpp-use-emplace %t

namespace std {

template <typename T>
class vector {
public:
  void push_back(const T &) {}
  void push_back(T &&) {}

  template <typename... Args>
  void emplace_back(Args &&... args){};
};
} // namespace std

class Foo {
public:
  Foo() : _num1(0)
  // CHECK-MESSAGES: warning: constructor does not initialize these fields: _num2 [cppcoreguidelines-pro-type-member-init,hicpp-member-init]
  {
    _num1 = 10;
  }

  int use_the_members() const {
    return _num1 + _num2;
  }

private:
  int _num1;
  int _num2;
  // CHECK-FIXES: int _num2{};
};

void should_use_emplace(std::vector<Foo> &v) {
  v.push_back(Foo());
  // CHECK-FIXES: v.emplace_back();
  // CHECK-MESSAGES: warning: use emplace_back instead of push_back [hicpp-use-emplace,modernize-use-emplace]
}

