#include "instrprof-dynamic-header.h"
void a() {                             // COV: [[@LINE]]| 1|void a
  if (true) {                          // COV: [[@LINE]]| 1|  if
    bar<void>(1);                      // COV: [[@LINE]]| 1|    bar
    bar<char>(1);                      // COV: [[@LINE]]| 1|   bar
  }                                    // COV: [[@LINE]]| 1|  }
}
