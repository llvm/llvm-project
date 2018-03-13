#include "foo.h"

struct Main {
  Main();
  int x;
};

Main::Main() : x(47) {
    bool some_code = x == 47; // !BR_main
}

Main MainObj;

int main() { return MainObj.x + FooObj.x; }
