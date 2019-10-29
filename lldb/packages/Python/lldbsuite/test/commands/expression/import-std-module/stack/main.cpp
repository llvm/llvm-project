#include <list>
#include <stack>
#include <vector>

struct C {
  // Constructor for testing emplace.
  C(int i) : i(i) {};
  int i;
};

int main(int argc, char **argv) {
  // std::deque is the default container.
  std::stack<C> s_deque({{1}, {2}, {3}});
  std::stack<C, std::vector<C>> s_vector({{1}, {2}, {3}});
  std::stack<C, std::list<C>> s_list({{1}, {2}, {3}});
  return 0; // Set break point at this line.
}
