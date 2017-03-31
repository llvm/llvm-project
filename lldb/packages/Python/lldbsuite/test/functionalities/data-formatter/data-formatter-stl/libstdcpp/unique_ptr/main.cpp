#include <memory>
#include <string>

struct Deleter {
  void operator()(void *) {}

  int a;
  int b;
};

struct Foo {
  int data;
  std::unique_ptr<Foo> fp;
};

int main() {
  std::unique_ptr<char> nup;
  std::unique_ptr<int> iup(new int{123});
  std::unique_ptr<std::string> sup(new std::string("foobar"));

  std::unique_ptr<char, Deleter> ndp;
  std::unique_ptr<int, Deleter> idp(new int{456}, Deleter{1, 2});
  std::unique_ptr<std::string, Deleter> sdp(new std::string("baz"),
                                            Deleter{3, 4});

  std::unique_ptr<Foo> fp(new Foo{3});

  // Set up a structure where we have a loop in the unique_ptr chain.
  Foo* f1 = new Foo{1};
  Foo* f2 = new Foo{2};
  f1->fp.reset(f2);
  f2->fp.reset(f1);

  return 0; // Set break point at this line.
}
