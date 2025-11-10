//===- map_elf_virtual_address_test.c -------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
  Status = amd_comgr_action_data_get_data(
      DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &DataExec);
  bool Nobits;
  uint64_t ElfVirtualAddress = 0x60;
  uint64_t CodeObjectOffset = -1;
  uint64_t SliceSize = -1;

  // phdr.p_vaddr:   0
  // phdr.p_vaddr + phdr.p_memsz:  0x8c0
  // phdr.p_offset:   0
  // phdr.p_filesz:  0x8c0
  // phdr.p_memsz:  0x8c0
  // codeObjectOffset == elfVirtualAddress - phdr.p_vaddr + phdr.p_offset
  // nobits = phdr.p_vaddr >= phdr.p_filesz
  // slizesize = phdr.p_memsz - (elfVirtualAddress - phdr.p_vaddr);
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(
      DataExec, ElfVirtualAddress, &CodeObjectOffset, &SliceSize, &Nobits);
  checkError(Status, "amd_comgr_map_elf_virtual_address_to_code_object_offset");

  if (CodeObjectOffset != 0x60 || Nobits != 0 || SliceSize != 0x860) {
    printf("elf virtual address map failed for address %#6" PRIx64 "\n"
           "  Expected: codeObjectOffset = 0x60, nobits = 0, slice = 0x\n"
           "  Actual:   codeObjectOffset = %#6" PRIx64
           ", nobits = %d, slice = %#6" PRIx64 "\n",
           ElfVirtualAddress, CodeObjectOffset, Nobits, SliceSize);
    exit(1);
  }

  ElfVirtualAddress = 0x1400;
  CodeObjectOffset = -1;
  // phdr.p_vaddr:   0x1000
  // phdr.p_vaddr + phdr.p_memsz:  0x1580
  // phdr.p_offset:   0x1000
  // phdr.p_filesz:  0x580
  // phdr.p_memsz:  0x580
  // codeObjectOffset == elfVirtualAddress - phdr.p_vaddr + phdr.p_offset
  // nobits = phdr.p_vaddr >= phdr.p_filesz
  // slizesize = phdr.p_memsz - (elfVirtualAddress - phdr.p_vaddr);
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(
      DataExec, ElfVirtualAddress, &CodeObjectOffset, &SliceSize, &Nobits);
  checkError(Status, "amd_comgr_map_elf_virtual_address_to_code_object_offset");

  if (CodeObjectOffset != 0x1400 || Nobits != 0 || SliceSize != 0x180) {
    printf("elf virtual address map failed for address %#6" PRIx64 "\n"
           "  Expected: codeObjectOffset = 0x1400, nobits = 0, slice = 0x180\n"
           "  Actual:   codeObjectOffset = %#6" PRIx64
           ", nobits = %d, slice = %#6" PRIx64 "\n",
           ElfVirtualAddress, CodeObjectOffset, Nobits, SliceSize);
    exit(1);
  }

  ElfVirtualAddress = 0x2035;
  CodeObjectOffset = -1;
  // phdr.p_vaddr:   0x2000
  // phdr.p_vaddr + phdr.p_memsz:  0x2070
  // phdr.p_offset:   0x2000
  // phdr.p_filesz:  0x70
  // phdr.p_memsz:  0x70
  // codeObjectOffset == elfVirtualAddress - phdr.p_vaddr + phdr.p_offset
  // nobits = phdr.p_vaddr >= phdr.p_filesz
  // slizesize = phdr.p_memsz - (elfVirtualAddress - phdr.p_vaddr);
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(
      DataExec, ElfVirtualAddress, &CodeObjectOffset, &SliceSize, &Nobits);
  checkError(Status, "amd_comgr_map_elf_virtual_address_to_code_object_offset");

  if (CodeObjectOffset != 0x2035 || Nobits != 0 || SliceSize != 0x3b) {
    printf("elf virtual address map failed for address %#6" PRIx64 "\n"
           "  Expected: codeObjectOffset = 0x2035, nobits = 0, slice = 0x3b\n"
           "  Actual:   codeObjectOffset = %#6" PRIx64
           ", nobits = %d, slice = %#6" PRIx64 "\n",
           ElfVirtualAddress, CodeObjectOffset, Nobits, SliceSize);
    exit(1);
  }

  ElfVirtualAddress = 0x9000;
  CodeObjectOffset = -1;
  // invalid elf virtual address
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(
      DataExec, ElfVirtualAddress, &CodeObjectOffset, &SliceSize, &Nobits);
  if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT) {
    printf("elf virtual address map succeded on invalid address:\n"
           "  Address = %#6" PRIx64 "\n"
           "  codeObjectOffset = %#6" PRIx64 "\n",
           ElfVirtualAddress, CodeObjectOffset);
    exit(1);
  }

  // Test rocm 5.7 elf virtual address mapping
  amd_comgr_data_t DataExec2;
  Status = amd_comgr_action_data_get_data(
      DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE, 1, &DataExec2);
  ElfVirtualAddress = 0x60;
  CodeObjectOffset = -1;
  // phdr.p_vaddr:   0
  // phdr.p_vaddr + phdr.p_memsz:  0x8c0
  // phdr.p_offset:   0
  // phdr.p_filesz:  0x8c0
  // phdr.p_memsz:  0x8c0
  // codeObjectOffset == elfVirtualAddress - phdr.p_vaddr + phdr.p_offset
  // nobits = phdr.p_vaddr >= phdr.p_filesz
  // slizesize = phdr.p_memsz - (elfVirtualAddress - phdr.p_vaddr);
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(
      DataExec2, ElfVirtualAddress, &CodeObjectOffset, &SliceSize, &Nobits);
  checkError(Status, "amd_comgr_map_elf_virtual_address_to_code_object_offset");

  if (CodeObjectOffset != 0x60 || Nobits != 0 || SliceSize != 0x860) {
    printf("elf virtual address map failed for address %#6" PRIx64 "\n"
           "  Expected: codeObjectOffset = 0x60, nobits = 0, slice = 0x860\n"
           "  Actual:   codeObjectOffset = %#6" PRIx64
           ", nobits = %d, slice = %#6" PRIx64 "\n",
           ElfVirtualAddress, CodeObjectOffset, Nobits, SliceSize);
    exit(1);
  }

  ElfVirtualAddress = 0x1a00;
  CodeObjectOffset = -1;
  // phdr.p_vaddr:   0x1900
  // phdr.p_vaddr + phdr.p_memsz:  0x1e80
  // phdr.p_offset:   0x900
  // phdr.p_filesz:  0x580
  // phdr.p_memsz:  0x580
  // codeObjectOffset == elfVirtualAddress - phdr.p_vaddr + phdr.p_offset
  // nobits = phdr.p_vaddr >= phdr.p_filesz
  // slizesize = phdr.p_memsz - (elfVirtualAddress - phdr.p_vaddr);
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(
      DataExec2, ElfVirtualAddress, &CodeObjectOffset, &SliceSize, &Nobits);
  checkError(Status, "amd_comgr_map_elf_virtual_address_to_code_object_offset");

  if (CodeObjectOffset != 0xa00 || Nobits != 0 || SliceSize != 0x480) {
    printf("elf virtual address map failed for address %#6" PRIx64 "\n"
           "  Expected: codeObjectOffset = 0xa00, nobits = 0, slice = 0x480\n"
           "  Actual:   codeObjectOffset = %#6" PRIx64
           ", nobits = %d, slice = %#6" PRIx64 "\n",
           ElfVirtualAddress, CodeObjectOffset, Nobits, SliceSize);
    exit(1);
  }

  ElfVirtualAddress = 0x2e90;
  CodeObjectOffset = -1;
  // phdr.p_vaddr:   0x2e80
  // phdr.p_vaddr + phdr.p_memsz:  0x2ef0
  // phdr.p_offset:   0xe80
  // phdr.p_filesz:  0x70
  // phdr.p_memsz:  0x70
  // codeObjectOffset == elfVirtualAddress - phdr.p_vaddr + phdr.p_offset
  // nobits = phdr.p_vaddr >= phdr.p_filesz
  // slizesize = phdr.p_memsz - (elfVirtualAddress - phdr.p_vaddr);
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(
      DataExec2, ElfVirtualAddress, &CodeObjectOffset, &SliceSize, &Nobits);
  checkError(Status, "amd_comgr_map_elf_virtual_address_to_code_object_offset");

  if (CodeObjectOffset != 0xe90 || Nobits != 0 || SliceSize != 0x60) {
    printf("elf virtual address map failed for address %#6" PRIx64 "\n"
           "  Expected: codeObjectOffset = 0x2035, nobits = 0, slice = 0x60\n"
           "  Actual:   codeObjectOffset = %#6" PRIx64
           ", nobits = %d, slice = %#6" PRIx64 "\n",
           ElfVirtualAddress, CodeObjectOffset, Nobits, SliceSize);
    exit(1);
  }

  ElfVirtualAddress = 0x9000;
  CodeObjectOffset = -1;
  // invalid elf virtual address
  Status = amd_comgr_map_elf_virtual_address_to_code_object_offset(
      DataExec, ElfVirtualAddress, &CodeObjectOffset, &SliceSize, &Nobits);
  if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT) {
    printf("elf virtual address map succeded on invalid address:\n"
           "  Address = %#6" PRIx64 "\n"
           "  codeObjectOffset = %#6" PRIx64 "\n",
           ElfVirtualAddress, CodeObjectOffset);
    exit(1);
  }

  Status = amd_comgr_release_data(DataSource1);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataSource2);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataExec);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataExec2);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_destroy_data_set(DataSetExec);
  checkError(Status, "amd_comgr_destroy_data_set");
  free(BufSource1);
  free(BufSource2);
}
