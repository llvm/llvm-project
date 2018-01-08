#include <queue>
#include <vector>

using namespace std;

int main() {
  queue<int> q1{{1,2,3,4,5}};
  queue<int, std::vector<int>> q2{{1,2,3,4,5}};
  int ret = q1.size() + q2.size(); // break here
  return ret;
}
