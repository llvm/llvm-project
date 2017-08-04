#include <memory>
#include <string>

struct Deleter {
  void operator()(void *) {}

  int a;
  int b;
};

int main() {
  std::unique_ptr<char> nup;
  std::unique_ptr<int> iup(new int{123});
  std::unique_ptr<std::string> sup(new std::string("foobar"));

  std::unique_ptr<char, Deleter> ndp;
  std::unique_ptr<int, Deleter> idp(new int{456}, Deleter{1, 2});
  std::unique_ptr<std::string, Deleter> sdp(new std::string("baz"),
                                            Deleter{3, 4});

  return 0; // Set break point at this line.
}
