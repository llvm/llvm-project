#include <memory>

int main(int argc, char **argv) {
  std::shared_ptr<int> s(new int);
  *s = 3;
  std::weak_ptr<int> w = s;
  return *s; // Set break point at this line.
}
