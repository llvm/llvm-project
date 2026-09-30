// RUN: %clangxx_tysan -O0 %s -o %t && %run %t

#include <vector>

struct Registry {
    std::vector<int> arr;
    char temp_byte;
    bool bool_var = false;
};

int main() {
    static Registry r;
    r.arr.push_back(0);
    r.bool_var = true;
}
