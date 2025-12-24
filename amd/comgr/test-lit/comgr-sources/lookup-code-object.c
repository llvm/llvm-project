//===- lookup-code-object.c -----------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int main(int argc, char *argv[]) {
  amd_comgr_data_kind_t Kind;
  switch(atoi(argv[2])) {
  case 0:
    Kind = AMD_COMGR_DATA_KIND_EXECUTABLE;
    break;
  case 1:
    Kind = AMD_COMGR_DATA_KIND_FATBIN;
  }

  char *BufObject;
  size_t SizeObject = setBuf(argv[1], &BufObject);

  amd_comgr_data_t DataObject;
  amd_comgr_(create_data(Kind, &DataObject));
  amd_comgr_(set_data(DataObject, SizeObject, BufObject));

  amd_comgr_code_object_info_t ObjectInfo[3];
  ObjectInfo[0].isa = "amdgcn-amd-amdhsa--gfx900";
  ObjectInfo[0].size = 0;
  ObjectInfo[0].offset = 0;

  ObjectInfo[1].isa = "amdgcn-amd-amdhsa--gfx942";
  ObjectInfo[1].size = 0;
  ObjectInfo[1].offset = 0;

  ObjectInfo[2].isa = "amdgcn-amd-amdhsa--gfx950";
  ObjectInfo[2].size = 0;
  ObjectInfo[2].offset = 0;

  amd_comgr_(lookup_code_object(DataObject, ObjectInfo, 3));

  for (int i = 0; i < 3; ++i) {
    printf("ObjectInfo[%d].isa: %s\n", i, ObjectInfo[i].isa);
    printf("ObjectInfo[%d].size: %ld\n", i, ObjectInfo[i].size);
    printf("ObjectInfo[%d].offset: %ld\n", i, ObjectInfo[i].offset);
  }

  return 0;
}
