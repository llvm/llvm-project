#include <stdio.h>
#include <stdexcept>

int twelve(int i) {
  return 12 + i; // break 12
}

int thirteen(int i) {
  return 13 + i; // break 13
}

namespace a {
  int fourteen(int i) {
    return 14 + i; // break 14
  }
}
int main(int argc, char const *argv[]) {
  for (int i=0; i<10; ++i) {
    int x = twelve(i) + thirteen(i) + a::fourteen(i); // break loop
  }
  try {
    throw std::invalid_argument( "throwing exception for testing" );
  } catch (...) {
    puts("caught exception...");
  }
  return 0;
}
