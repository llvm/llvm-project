#include <assert.h>
#include <libunwind.h>

int main(int, char**) {
  unw_context_t context;
  int ret = unw_getcontext(&context);
  assert(ret == UNW_ESUCCESS);
  return 0;
}
