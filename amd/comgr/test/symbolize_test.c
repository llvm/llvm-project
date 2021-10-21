/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct container {
  char *data;
  int sz;
} container_t;

void print_symbolized_string(const char *input, void *data) {
  int sz = strlen(input);
  container_t *ptr = (container_t *)data;
  ptr->data = (char *)malloc(sz + 1);
  ptr->data[sz] = '\0';
  memcpy(ptr->data, input, sz);
  printf("\n symbolized string is %s", ptr->data);
  free(ptr->data);
}

int main(int argc, char *argv[]) {
  size_t Size1;
  char *Buf1;
  amd_comgr_data_t DataIn1;
  amd_comgr_status_t Status;
  amd_comgr_symbolizer_info_t symbolizer;
  container_t user_data;

  // Read input file
  Size1 = setBuf(TEST_OBJ_DIR "/shared.so", &Buf1);

  // Create data object
  {
    printf("Test create input data set\n");

    Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataIn1);
    checkError(Status, "amd_comgr_create_data");
    Status = amd_comgr_set_data(DataIn1, Size1, Buf1);
    checkError(Status, "amd_comgr_set_data");
    Status = amd_comgr_set_data_name(DataIn1, "DO_IN1");
    checkError(Status, "amd_comgr_set_data_name");
  }

  {
    printf("Test create symbolizer info\n");

    Status = amd_comgr_create_symbolizer_info(DataIn1, &print_symbolized_string,
                                              &symbolizer);
    checkError(Status, "amd_comgr_create_symbolizer_info");
    Status = amd_comgr_symbolize(symbolizer, 100, 1, (void *)&user_data);
    checkError(Status, "amd_comgr_symbolize");
  }

  {
    printf("Test destroy symbolizer info\n");

    Status = amd_comgr_destroy_symbolizer_info(symbolizer);
    checkError(Status, "amd_comgr_destroy_symbolizer_info");
    Status = amd_comgr_release_data(DataIn1);
    checkError(Status, "amd_comgr_release_data");
    free(Buf1);
  }

  return 0;
}
