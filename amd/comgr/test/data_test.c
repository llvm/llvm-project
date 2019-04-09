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

int main(int argc, char *argv[]) {
  long size1;
  char *buf;
  amd_comgr_data_t dataObject, dataObject2, dataObject3;
  amd_comgr_data_set_t dataSet;
  amd_comgr_status_t status;
  size_t count;

  // Read input file
  size1 = setBuf(TEST_OBJ_DIR "/shared.so", &buf);

  // Create data object
  {
    printf("Test 1 ...\n");

    status =
        amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &dataObject);
    checkError(status, "amd_comgr_create_data");

    status = amd_comgr_set_data(dataObject, size1, buf);
    checkError(status, "amd_comgr_set_data");
  }

  {
    printf("Test 2 ...\n");
    status = amd_comgr_set_data_name(dataObject, "DO1");
    checkError(status, "amd_comgr_set_data_name");

    size_t size;
    char name[10];
    status = amd_comgr_get_data_name(dataObject, &size, NULL);
    checkError(status, "amd_comgr_get_data_name");
    if (size != strlen("DO1") + 1) {
      printf("FAILED_2a:\n");
      printf("  amd_comgr_get_data_name size = %ld\n", size);
      printf("  expected size = %ld\n", strlen("DO1"));
    }
    status = amd_comgr_get_data_name(dataObject, &size, &name[0]);
    checkError(status, "amd_comgr_get_data_name");
    if (strcmp(name, "DO1")) {
      printf("FAILED_2b:\n");
      printf("   amd_comgr_get_data_name name = %s\n", &name[0]);
      printf("   expected name = DO1\n");
    }
  }

  {
    printf("Test 3 ...\n");

    // Add data object 1
    status = amd_comgr_create_data_set(&dataSet);
    checkError(status, "amd_cogmr_create_data_set");

    // Add data object
    status = amd_comgr_data_set_add(dataSet, dataObject);
    checkError(status, "amd_cogmr_data_set_add");

    // Add data object 2
    status =
        amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &dataObject2);
    checkError(status, "amd_comgr_create_data_2");
    status = amd_comgr_set_data(dataObject2, size1, buf); // Use the same data
    checkError(status, "amd_comgr_set_data_2");
    status = amd_comgr_set_data_name(dataObject2, "DO2");
    checkError(status, "amd_comgr_set_data_name_2");
    status = amd_comgr_data_set_add(dataSet, dataObject2);
    checkError(status, "amd_cogmr_data_set_add_2");

    // Add data object 3
    status =
        amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &dataObject3);
    checkError(status, "amd_comgr_create_data_3");
    status = amd_comgr_set_data(dataObject3, size1, buf); // Use the same data
    checkError(status, "amd_comgr_set_data_3");
    status = amd_comgr_set_data_name(dataObject3, "DO3");
    checkError(status, "amd_comgr_set_data_name_3");
    status = amd_comgr_data_set_add(dataSet, dataObject3);
    checkError(status, "amd_cogmr_data_set_add_3");

    status = amd_comgr_action_data_count(
        dataSet, AMD_COMGR_DATA_KIND_RELOCATABLE, &count);
    checkError(status, "amd_comgr_action_data_count");
    if (count != 3) {
      printf("FAILED_3a:\n");
      printf("   amd_comgr_action_data_count = %ld\n", count);
      printf("   expected count = 3\n");
    }

    amd_comgr_data_t data2;
    status = amd_comgr_action_data_get_data(
        dataSet, AMD_COMGR_DATA_KIND_RELOCATABLE, 2, &data2);
    checkError(status, "amd_comgr_action_data_get_data");
    size_t size2;
    char name2[10];
    status = amd_comgr_get_data_name(data2, &size2, NULL);
    checkError(status, "amd_comgr_get_data_name");
    status = amd_comgr_get_data_name(data2, &size2, &name2[0]);
    if (strcmp(name2, "DO3")) {
      printf("FAILED_3b:\n");
      printf("   amd_comgr_get_data_name name_2 = %s\n", &name2[0]);
      printf("   expected name = DO2\n");
    }

    // dataObject1, dataObject2 has refcount = 2, dataObject3 has refcount = 3.
    amd_comgr_release_data(data2);
    // dataObject1, dataObject2 has refcount = 2, dataObject3 has refcount = 2.
  }

  {
    printf("Test 4 ...\n");

    // Remove data object.
    status = amd_comgr_data_set_remove(dataSet, AMD_COMGR_DATA_KIND_EXECUTABLE);
    checkError(status, "amd_cogmr_data_set_remove"); // nothing to remove
    status = amd_comgr_action_data_count(
        dataSet, AMD_COMGR_DATA_KIND_RELOCATABLE, &count);
    checkError(status, "amd_comgr_action_data_count");
    if (count != 3) {
      printf("FAILED_4a:\n");
      printf("   amd_comgr_action_data_count = %ld\n", count);
      printf("   expected count = 3\n");
    }

    status =
        amd_comgr_data_set_remove(dataSet, AMD_COMGR_DATA_KIND_RELOCATABLE);
    checkError(status, "amd_cogmr_data_set_remove_2");
    status = amd_comgr_action_data_count(
        dataSet, AMD_COMGR_DATA_KIND_RELOCATABLE, &count);
    checkError(status, "amd_comgr_action_data_count");
    if (count != 0) {
      printf("FAILED_4b:\n");
      printf("   amd_comgr_action_data_count = %ld\n", count);
      printf("   expected count = 1\n");
    }

    // dataObject1, dataObject2 has refcount = 1, dataObject3 has refcount = 1.

    amd_comgr_data_kind_t kind2;
    status = amd_comgr_get_data_kind(dataObject, &kind2);
    checkError(status, "amd_cogmr_get_data_kind");
    if (kind2 != AMD_COMGR_DATA_KIND_RELOCATABLE) {
      printf("FAILED_4c:\n");
      printf("  amd_comgr_get_data_kind kind = %d\n", kind2);
    }

    // insert 3 items back into set
    status = amd_comgr_data_set_add(dataSet, dataObject);
    status = amd_comgr_data_set_add(dataSet, dataObject2);
    status = amd_comgr_data_set_add(dataSet, dataObject3);

    // Destroy data set, amd_comgr_release_data to be called also
    status = amd_comgr_destroy_data_set(dataSet);
    checkError(status, "amd_comgr_destroy_data_set");
  }

  {
    printf("Cleanup ...\n");
    status = amd_comgr_release_data(dataObject);
    checkError(status, "amd_comgr_release_data");
    status = amd_comgr_release_data(dataObject2);
    checkError(status, "amd_comgr_release_data");
    status = amd_comgr_release_data(dataObject3);
    checkError(status, "amd_comgr_release_data");
    free(buf);
  }

  return 0;
}
