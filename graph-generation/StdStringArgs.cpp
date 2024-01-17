#include<string>

void f(const std::string& s) { }

void StdStringArgumentCall(const std::string& s) {
  f(s.c_str());
}
