#include "amdocl.h"

kernel void test_get_local_id_0(global uint* out)
{
  out[0] = get_local_id(0);
}

