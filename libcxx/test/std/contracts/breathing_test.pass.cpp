#include <cassert>
#include <contracts>

using std::contracts::contract_violation;

void test(int x) pre(x != 1) {
  contract_assert(x != 0);
}


int main() {
  test(2);
  test(1);
}
