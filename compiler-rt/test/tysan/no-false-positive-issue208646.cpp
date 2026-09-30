// RUN: %clangxx_tysan -O0 %s -o %t && %run %t

#include <string>
#include <optional>

static std::optional<std::string> optional_var = std::nullopt;

int main() { 
  optional_var = "this is a random long string (short one does not reproduce)";
  return 0;
}
