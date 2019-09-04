@import Foundation;
#include "Foo.h"
#include "Bar.h"

int main(int argc, char **argv) {
  [[[Foo alloc] init] f];
  [[[Bar alloc] init] f];
  return 0;
}
