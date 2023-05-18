#include "instrprof-dynamic-header.h"
void b() {
  if (true) {
    bar<void>(1);
    bar<int>(1);
  }
}
