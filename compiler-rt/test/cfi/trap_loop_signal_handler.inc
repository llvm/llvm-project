#include <sanitizer/ubsan_interface.h>

static __attribute__((constructor)) void install_trap_loop_detection() {
  __ubsan_install_trap_loop_detection();
}
