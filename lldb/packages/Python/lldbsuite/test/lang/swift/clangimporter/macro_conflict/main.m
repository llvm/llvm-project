@import Foundation;
#include "Foo.h"
#include "Bar.h"

int main(int argc, char **argv) {
  [[[Bar alloc] init] f];
  [[[Foo alloc] init] f];
  return 0;
}
