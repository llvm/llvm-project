#include <string>
#include <vector>

struct bar {
  int var1;
};

struct foo {
  int var1;
  bar* my_bar_pointer;
  bar my_bar_object;
  foo* next_foo;
};

int fun(std::vector<std::string> var) {
  return var.size(); // breakpoint 1
}

int main(int argc, char const *argv[]) {
  int var1 = 0;
  int var2 = 1;
  std::string str1 = "a";
  std::string str2 = "b";
  std::vector<std::string> vec;
  fun(vec);
  bar bar1 = {2};
  bar* bar2 = &bar1; 
  foo foo1 = {3,&bar1, bar1, NULL};
  return 0; // breakpoint 2
}
