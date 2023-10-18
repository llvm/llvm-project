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
  char *BufSource1, *BufSource2;
  size_t SizeSource1, SizeSource2;
  amd_comgr_data_t DataSource1, DataSource2;
  amd_comgr_data_set_t DataSetExec;
  amd_comgr_status_t Status;

  // TODO: We need to add the source code for these objects to the
  // repository. We should also update them to include some headers
  // in a nobits segment
  SizeSource1 = setBuf(TEST_OBJ_DIR "/rocm56slice.b", &BufSource1);
  SizeSource2 = setBuf(TEST_OBJ_DIR "/rocm57slice.b", &BufSource2);

  Status = amd_comgr_create_data_set(&DataSetExec);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataSource1);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource1, SizeSource1, BufSource1);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource1, "rocm56slice.b");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetExec, DataSource1);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataSource2);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource2, SizeSource2, BufSource2);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource2, "rocm57slice.b");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetExec, DataSource2);
  checkError(Status, "amd_comgr_data_set_add");

  size_t Count;
  Status = amd_comgr_action_data_count(DataSetExec,
                                       AMD_COMGR_DATA_KIND_EXECUTABLE, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 2) {
    printf("Creating executable data set failed: "
           "produced %zu executable objects (expected 2)\n",
           Count);
    exit(1);
  }

  // Test rocm 5.6 elf virtual address mapping
  amd_comgr_data_t DataExec;
  Status = amd_comgr_action_data_get_data(DataSetExec,
                                          AMD_COMGR_DATA_KIND_EXECUTABLE,
                                          0, &DataExec);
  bool nobits;
  uint64_t elfVirtualAddress = 0x60;
  uint64_t codeObjectOffset = -1;
  uint64_t sliceSize = -1;

  // phdr.p_vaddr:   0
  // phdr.p_vaddr + phdr.p_memsz:  0x8c0
  // phdr.p_offset:   0
  // phdr.p_filesz:  0x8c0
  // phdr.p_memsz:  0x8c0
  // codeObjectOffset == elfVirtualAddress - phdr.p_vaddr + phdr.p_offset
  // nobits = phdr.p_vaddr >= phdr.p_filesz
  // slizesize = phdr.p_memsz - (elfVirtualAddress - phdr.p_vaddr);
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(DataExec,
                                                              elfVirtualAddress,
                                                              &codeObjectOffset,
                                                              &sliceSize,
                                                              &nobits);
  checkError(Status, "amd_comgr_map_elf_virtual_address_to_code_object_offset");

  if (codeObjectOffset != 0x60 || nobits != 0 || sliceSize != 0x860) {
    printf("elf virtual address map failed for address %#6lx\n"
           "  Expected: codeObjectOffset = 0x60, nobits = 0, slice = 0x\n"
           "  Actual:   codeObjectOffset = %#6lx, nobits = %d, slice = %#6lx\n",
           elfVirtualAddress, codeObjectOffset, nobits, sliceSize);
    exit(1);
  }

  elfVirtualAddress = 0x1400;
  codeObjectOffset = -1;
  // phdr.p_vaddr:   0x1000
  // phdr.p_vaddr + phdr.p_memsz:  0x1580
  // phdr.p_offset:   0x1000
  // phdr.p_filesz:  0x580
  // phdr.p_memsz:  0x580
  // codeObjectOffset == elfVirtualAddress - phdr.p_vaddr + phdr.p_offset
  // nobits = phdr.p_vaddr >= phdr.p_filesz
  // slizesize = phdr.p_memsz - (elfVirtualAddress - phdr.p_vaddr);
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(DataExec,
                                                              elfVirtualAddress,
                                                              &codeObjectOffset,
                                                              &sliceSize,
                                                              &nobits);
  checkError(Status, "amd_comgr_map_elf_virtual_address_to_code_object_offset");

  if (codeObjectOffset != 0x1400 || nobits != 0 || sliceSize != 0x180) {
    printf("elf virtual address map failed for address %#6lx\n"
           "  Expected: codeObjectOffset = 0x1400, nobits = 0, slice = 0x180\n"
           "  Actual:   codeObjectOffset = %#6lx, nobits = %d, slice = %#6lx\n",
           elfVirtualAddress, codeObjectOffset, nobits, sliceSize);
    exit(1);
  }

  elfVirtualAddress = 0x2035;
  codeObjectOffset = -1;
  // phdr.p_vaddr:   0x2000
  // phdr.p_vaddr + phdr.p_memsz:  0x2070
  // phdr.p_offset:   0x2000
  // phdr.p_filesz:  0x70
  // phdr.p_memsz:  0x70
  // codeObjectOffset == elfVirtualAddress - phdr.p_vaddr + phdr.p_offset
  // nobits = phdr.p_vaddr >= phdr.p_filesz
  // slizesize = phdr.p_memsz - (elfVirtualAddress - phdr.p_vaddr);
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(DataExec,
                                                              elfVirtualAddress,
                                                              &codeObjectOffset,
                                                              &sliceSize,
                                                              &nobits);
  checkError(Status, "amd_comgr_map_elf_virtual_address_to_code_object_offset");

  if (codeObjectOffset != 0x2035 || nobits != 0 || sliceSize != 0x3b) {
    printf("elf virtual address map failed for address %#6lx\n"
           "  Expected: codeObjectOffset = 0x2035, nobits = 0, slice = 0x3b\n"
           "  Actual:   codeObjectOffset = %#6lx, nobits = %d, slice = %#6lx\n",
           elfVirtualAddress, codeObjectOffset, nobits, sliceSize);
    exit(1);
  }

  elfVirtualAddress = 0x9000;
  codeObjectOffset = -1;
  // invalid elf virtual address
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(DataExec,
                                                              elfVirtualAddress,
                                                              &codeObjectOffset,
                                                              &sliceSize,
                                                              &nobits);
  if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT) {
    printf("elf virtual address map succeded on invalid address:\n"
           "  Address = %#6lx\n"
           "  codeObjectOffset = %#6lx\n",
           elfVirtualAddress, codeObjectOffset);
    exit(1);
  }

  // Test rocm 5.7 elf virtual address mapping
  Status = amd_comgr_action_data_get_data(DataSetExec,
                                          AMD_COMGR_DATA_KIND_EXECUTABLE,
                                          1, &DataExec);
  elfVirtualAddress = 0x60;
  codeObjectOffset = -1;
  // phdr.p_vaddr:   0
  // phdr.p_vaddr + phdr.p_memsz:  0x8c0
  // phdr.p_offset:   0
  // phdr.p_filesz:  0x8c0
  // phdr.p_memsz:  0x8c0
  // codeObjectOffset == elfVirtualAddress - phdr.p_vaddr + phdr.p_offset
  // nobits = phdr.p_vaddr >= phdr.p_filesz
  // slizesize = phdr.p_memsz - (elfVirtualAddress - phdr.p_vaddr);
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(DataExec,
                                                              elfVirtualAddress,
                                                              &codeObjectOffset,
                                                              &sliceSize,
                                                              &nobits);
  checkError(Status, "amd_comgr_map_elf_virtual_address_to_code_object_offset");

  if (codeObjectOffset != 0x60 || nobits != 0 || sliceSize != 0x860) {
    printf("elf virtual address map failed for address %#6lx\n"
           "  Expected: codeObjectOffset = 0x60, nobits = 0, slice = 0x860\n"
           "  Actual:   codeObjectOffset = %#6lx, nobits = %d, slice = %#6lx\n",
           elfVirtualAddress, codeObjectOffset, nobits, sliceSize);
    exit(1);
  }

  elfVirtualAddress = 0x1a00;
  codeObjectOffset = -1;
  // phdr.p_vaddr:   0x1900
  // phdr.p_vaddr + phdr.p_memsz:  0x1e80
  // phdr.p_offset:   0x900
  // phdr.p_filesz:  0x580
  // phdr.p_memsz:  0x580
  // codeObjectOffset == elfVirtualAddress - phdr.p_vaddr + phdr.p_offset
  // nobits = phdr.p_vaddr >= phdr.p_filesz
  // slizesize = phdr.p_memsz - (elfVirtualAddress - phdr.p_vaddr);
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(DataExec,
                                                              elfVirtualAddress,
                                                              &codeObjectOffset,
                                                              &sliceSize,
                                                              &nobits);
  checkError(Status, "amd_comgr_map_elf_virtual_address_to_code_object_offset");

  if (codeObjectOffset != 0xa00 || nobits != 0 || sliceSize != 0x480) {
    printf("elf virtual address map failed for address %#6lx\n"
           "  Expected: codeObjectOffset = 0xa00, nobits = 0, slice = 0x480\n"
           "  Actual:   codeObjectOffset = %#6lx, nobits = %d, slice = %#6lx\n",
           elfVirtualAddress, codeObjectOffset, nobits, sliceSize);
    exit(1);
  }

  elfVirtualAddress = 0x2e90;
  codeObjectOffset = -1;
  // phdr.p_vaddr:   0x2e80
  // phdr.p_vaddr + phdr.p_memsz:  0x2ef0
  // phdr.p_offset:   0xe80
  // phdr.p_filesz:  0x70
  // phdr.p_memsz:  0x70
  // codeObjectOffset == elfVirtualAddress - phdr.p_vaddr + phdr.p_offset
  // nobits = phdr.p_vaddr >= phdr.p_filesz
  // slizesize = phdr.p_memsz - (elfVirtualAddress - phdr.p_vaddr);
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(DataExec,
                                                              elfVirtualAddress,
                                                              &codeObjectOffset,
                                                              &sliceSize,
                                                              &nobits);
  checkError(Status, "amd_comgr_map_elf_virtual_address_to_code_object_offset");

  if (codeObjectOffset != 0xe90 || nobits != 0 || sliceSize != 0x60) {
    printf("elf virtual address map failed for address %#6lx\n"
           "  Expected: codeObjectOffset = 0x2035, nobits = 0, slice = 0x60\n"
           "  Actual:   codeObjectOffset = %#6lx, nobits = %d, slice = %#6lx\n",
           elfVirtualAddress, codeObjectOffset, nobits, sliceSize);
    exit(1);
  }

  elfVirtualAddress = 0x9000;
  codeObjectOffset = -1;
  // invalid elf virtual address
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(DataExec,
                                                              elfVirtualAddress,
                                                              &codeObjectOffset,
                                                              &sliceSize,
                                                              &nobits);
  if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT) {
    printf("elf virtual address map succeded on invalid address:\n"
           "  Address = %#6lx\n"
           "  codeObjectOffset = %#6lx\n",
           elfVirtualAddress, codeObjectOffset);
    exit(1);
  }

  Status = amd_comgr_release_data(DataSource1);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataSource2);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataExec);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_destroy_data_set(DataSetExec);
  checkError(Status, "amd_comgr_destroy_data_set");
  free(BufSource1);
  free(BufSource2);
}
