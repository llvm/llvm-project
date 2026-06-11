#include "a.h"
#include "b.h"
int main() {
  int x = a_entry("wrong");   // error -> forces resolving A's shared.h location
  int y = b_entry("wrong");   // error -> forces resolving B's shared.h location
  return x + y;
}
