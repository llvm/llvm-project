#include <iostream>

void input_copy_loop() {
  std::string str;
  while (std::getline(std::cin, str))
    std::cout << "read: " << str << std::endl;
}

int main() {
  input_copy_loop();
  return 0;
}
