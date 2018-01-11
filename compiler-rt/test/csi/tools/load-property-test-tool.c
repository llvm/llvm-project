#include <stdlib.h>
#include <stdio.h>
#include "csi.h"

static int num_loads = 0, num_read_before_writes = 0;

void report() {
    printf("num_loads = %d\n", num_loads);
    printf("num_read_before_writes = %d\n", num_read_before_writes);
}

void __csi_init() {
    num_loads = num_read_before_writes = 0;
    atexit(report);
}

void __csi_before_load(const csi_id_t load_id, const void *addr,
                       const int32_t num_bytes, const load_prop_t prop) {
    num_loads++;
    if (prop.is_read_before_write_in_bb) num_read_before_writes++;
}
