#include <utility>
#include <string>
#include <map>
#include <unordered_map>

int main(int argc, char **argv) {
  std::pair<int, long> pair = std::make_pair<int, float>(1, 2L);
  std::string foo = "bar";
  std::map<int, int> map;
  map[1] = 2;
  std::unordered_map<int, int> umap;
  umap[1] = 2;
  return pair.first; // Set break point at this line.
}
