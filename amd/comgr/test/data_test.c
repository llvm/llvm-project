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
  long Size1;
  char *Buf;
  amd_comgr_data_t DataObject, DataObject2, DataObject3;
  amd_comgr_data_set_t DataSet;
  amd_comgr_status_t Status;
  size_t Count;

  // Read input file
  Size1 = setBuf(TEST_OBJ_DIR "/shared.so", &Buf);

  // Create data object
  {
    printf("Test 1 ...\n");

    Status =
        amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &DataObject);
    checkError(Status, "amd_comgr_create_data");

    Status = amd_comgr_set_data(DataObject, Size1, Buf);
    checkError(Status, "amd_comgr_set_data");
  }

  {
    printf("Test 2 ...\n");
    Status = amd_comgr_set_data_name(DataObject, "DO1");
    checkError(Status, "amd_comgr_set_data_name");

    size_t Size;
    char Name[10];
    Status = amd_comgr_get_data_name(DataObject, &Size, NULL);
    checkError(Status, "amd_comgr_get_data_name");
    if (Size != strlen("DO1") + 1) {
      printf("FAILED_2a:\n");
      printf("  amd_comgr_get_data_name size = %zd\n", Size);
      printf("  expected size = %zd\n", strlen("DO1"));
    }
    Status = amd_comgr_get_data_name(DataObject, &Size, &Name[0]);
    checkError(Status, "amd_comgr_get_data_name");
    if (strcmp(Name, "DO1")) {
      printf("FAILED_2b:\n");
      printf("   amd_comgr_get_data_name name = %s\n", &Name[0]);
      printf("   expected name = DO1\n");
    }
  }

  {
    printf("Test 3 ...\n");

    // Add data object 1
    Status = amd_comgr_create_data_set(&DataSet);
    checkError(Status, "amd_cogmr_create_data_set");

    // Add data object
    Status = amd_comgr_data_set_add(DataSet, DataObject);
    checkError(Status, "amd_cogmr_data_set_add");

    // Add data object 2
    Status =
        amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &DataObject2);
    checkError(Status, "amd_comgr_create_data_2");
    Status = amd_comgr_set_data(DataObject2, Size1, Buf); // Use the same data
    checkError(Status, "amd_comgr_set_data_2");
    Status = amd_comgr_set_data_name(DataObject2, "DO2");
    checkError(Status, "amd_comgr_set_data_name_2");
    Status = amd_comgr_data_set_add(DataSet, DataObject2);
    checkError(Status, "amd_cogmr_data_set_add_2");

    // Add data object 3
    Status =
        amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &DataObject3);
    checkError(Status, "amd_comgr_create_data_3");
    Status = amd_comgr_set_data(DataObject3, Size1, Buf); // Use the same data
    checkError(Status, "amd_comgr_set_data_3");
    Status = amd_comgr_set_data_name(DataObject3, "DO3");
    checkError(Status, "amd_comgr_set_data_name_3");
    Status = amd_comgr_data_set_add(DataSet, DataObject3);
    checkError(Status, "amd_cogmr_data_set_add_3");

    Status = amd_comgr_action_data_count(
        DataSet, AMD_COMGR_DATA_KIND_RELOCATABLE, &Count);
    checkError(Status, "amd_comgr_action_data_count");
    if (Count != 3) {
      printf("FAILED_3a:\n");
      printf("   amd_comgr_action_data_count = %zd\n", Count);
      printf("   expected count = 3\n");
    }

    amd_comgr_data_t Data2;
    Status = amd_comgr_action_data_get_data(
        DataSet, AMD_COMGR_DATA_KIND_RELOCATABLE, 2, &Data2);
    checkError(Status, "amd_comgr_action_data_get_data");
    size_t Size2;
    char Name2[10];
    Status = amd_comgr_get_data_name(Data2, &Size2, NULL);
    checkError(Status, "amd_comgr_get_data_name");
    Status = amd_comgr_get_data_name(Data2, &Size2, &Name2[0]);
    if (strcmp(Name2, "DO3")) {
      printf("FAILED_3b:\n");
      printf("   amd_comgr_get_data_name name_2 = %s\n", &Name2[0]);
      printf("   expected name = DO2\n");
    }

    // dataObject1, dataObject2 has refcount = 2, dataObject3 has refcount = 3.
    amd_comgr_release_data(Data2);
    // dataObject1, dataObject2 has refcount = 2, dataObject3 has refcount = 2.
  }

  {
    printf("Test 4 ...\n");

    // Remove data object.
    Status = amd_comgr_data_set_remove(DataSet, AMD_COMGR_DATA_KIND_EXECUTABLE);
    checkError(Status, "amd_cogmr_data_set_remove"); // nothing to remove
    Status = amd_comgr_action_data_count(
        DataSet, AMD_COMGR_DATA_KIND_RELOCATABLE, &Count);
    checkError(Status, "amd_comgr_action_data_count");
    if (Count != 3) {
      printf("FAILED_4a:\n");
      printf("   amd_comgr_action_data_count = %zd\n", Count);
      printf("   expected count = 3\n");
    }

    Status =
        amd_comgr_data_set_remove(DataSet, AMD_COMGR_DATA_KIND_RELOCATABLE);
    checkError(Status, "amd_cogmr_data_set_remove_2");
    Status = amd_comgr_action_data_count(
        DataSet, AMD_COMGR_DATA_KIND_RELOCATABLE, &Count);
    checkError(Status, "amd_comgr_action_data_count");
    if (Count != 0) {
      printf("FAILED_4b:\n");
      printf("   amd_comgr_action_data_count = %zd\n", Count);
      printf("   expected count = 1\n");
    }

    // dataObject1, dataObject2 has refcount = 1, dataObject3 has refcount = 1.

    amd_comgr_data_kind_t Kind2;
    Status = amd_comgr_get_data_kind(DataObject, &Kind2);
    checkError(Status, "amd_cogmr_get_data_kind");
    if (Kind2 != AMD_COMGR_DATA_KIND_RELOCATABLE) {
      printf("FAILED_4c:\n");
      printf("  amd_comgr_get_data_kind kind = %d\n", Kind2);
    }

    // insert 3 items back into set
    Status = amd_comgr_data_set_add(DataSet, DataObject);
    Status = amd_comgr_data_set_add(DataSet, DataObject2);
    Status = amd_comgr_data_set_add(DataSet, DataObject3);

    // Destroy data set, amd_comgr_release_data to be called also
    Status = amd_comgr_destroy_data_set(DataSet);
    checkError(Status, "amd_comgr_destroy_data_set");
  }

  {
    printf("Cleanup ...\n");
    Status = amd_comgr_release_data(DataObject);
    checkError(Status, "amd_comgr_release_data");
    Status = amd_comgr_release_data(DataObject2);
    checkError(Status, "amd_comgr_release_data");
    Status = amd_comgr_release_data(DataObject3);
    checkError(Status, "amd_comgr_release_data");
    free(Buf);
  }

  return 0;
}
