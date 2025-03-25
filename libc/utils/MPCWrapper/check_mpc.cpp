#include <mpc.h>

int main() {
  mpc_t x;
  mpc_init2(x, 256);
  mpc_clear(x);
  return 0;
}
