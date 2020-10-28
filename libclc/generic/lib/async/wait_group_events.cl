#include <clc/clc.h>

_CLC_DEF _CLC_OVERLOAD void wait_group_events(int num_events,
                                              event_t *event_list) {
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
