#include <assert.h>
#include <libunwind.h>

int main() {
  unw_context_t context;
  int ret = unw_getcontext(&context);
  assert(ret == UNW_ESUCCESS);
}
