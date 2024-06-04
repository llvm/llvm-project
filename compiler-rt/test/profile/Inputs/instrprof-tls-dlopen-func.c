#include <stdint.h>

int8_t func(int8_t input) {
  if (input < 0) {
    return input;
  } else {
    return -input;
  }
}
