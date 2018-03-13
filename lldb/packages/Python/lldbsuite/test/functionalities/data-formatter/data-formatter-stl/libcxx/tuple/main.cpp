#include <tuple>
#include <string>

using namespace std;

int main() {
  tuple<> empty;
  tuple<int> one_elt{47};
  tuple<int, long, string> three_elts{1, 47l, "foo"};
  return 0; // break here
}
