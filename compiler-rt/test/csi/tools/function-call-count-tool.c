#include <stdlib.h>
#include <stdio.h>
#include "csi.h"

static int num_function_calls = 0;

void report() {
    printf("num_function_calls = %d\n", num_function_calls);
}

void __csi_init() {
    num_function_calls = 0;
    atexit(report);
}

void __csi_before_call(const csi_id_t call_id, const csi_id_t func_id,
                       const call_prop_t prop) {
    num_function_calls++;
}
