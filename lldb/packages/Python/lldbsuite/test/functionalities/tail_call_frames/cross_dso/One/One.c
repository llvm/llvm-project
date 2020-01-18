#include "shared.h"

__attribute__((noinline))
static void helper_in_a() {
  tail_called_in_b_from_a();
}

__attribute__((disable_tail_calls))
void tail_called_in_a_from_main() {
  helper_in_a();
}
