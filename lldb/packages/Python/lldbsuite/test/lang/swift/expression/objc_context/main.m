@import Foundation;
@import Bar;
#include "Foo.h"

@implementation Bar
@end

int main(int argc, char **argv) {
  [[Bar alloc] init];
  [[Foo alloc] init];
  return 0; // break here
}
