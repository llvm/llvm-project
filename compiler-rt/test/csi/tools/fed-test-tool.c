#include <stdlib.h>
#include <stdio.h>
#include "csi.h"

void __csi_func_entry(const csi_id_t func_id, const func_prop_t prop) {
    printf("Enter function %ld [%s:%d]\n", func_id,
           __csi_get_func_source_loc(func_id)->filename,
           __csi_get_func_source_loc(func_id)->line_number);
}
