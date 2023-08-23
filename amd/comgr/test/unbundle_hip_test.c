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
 *******************************************************************************/

/// -------
//  Manual recreation of Comgr bundle linking
//
//    // Create bundles
//    clang -c --offload-arch=gfx906 -emit-llvm -fgpu-rdc \
//    --gpu-bundle-output square.hip double.hip cube.hip
//
//    llvm-ar rc cube.a cube.bc
//
//    // Manually unbundle
//    clang-offload-bundler -type=bc \
//    -targets=hip-amdgcn-amd-amdhsa-gfx906 \
//    -input=square.bc -output=square-gfx906.bc \
//    -unbundle -allow-missing-bundles
//
//    clang-offload-bundler -type=bc \
//    -targets=hip-amdgcn-amd-amdhsa-gfx906 \
//    -input=double.bc -output=double-gfx906.bc \
//    -unbundle -allow-missing-bundles
//
//    clang-offload-bundler -type=a \
//    -targets=hip-amdgcn-amd-amdhsa-gfx906 \
//    -input=cube.a -output=cube-gfx906.a \
//    -unbundle -allow-missing-bundles \
//    -hip-openmp-compatible
//
//    // Manually link
//    llvm-link square-gfx906.bc double-gfx906.bc cube-gfx906.a \
//    -o gold/gold-linked-bitcode-gfx906.bc

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int Argc, char *Argv[]) {
  char *BufArchive, *BufBitcode1, *BufBitcode2;
  size_t SizeArchive, SizeBitcode1, SizeBitcode2;
  amd_comgr_data_t DataArchive, DataBitcode1, DataBitcode2;
  amd_comgr_data_set_t DataSetBundled, DataSetLinked, DataSetReloc,
                       DataSetExec;
  amd_comgr_action_info_t ActionInfo;
  amd_comgr_status_t Status;

  const char *IsaName = "amdgcn-amd-amdhsa--gfx906";

  SizeBitcode1 = setBuf("./source/square.bc", &BufBitcode1);
  SizeBitcode2 = setBuf("./source/double.bc", &BufBitcode2);
  SizeArchive = setBuf("./source/cube.a", &BufArchive);

  // Create Bundled dataset
  Status = amd_comgr_create_data_set(&DataSetBundled);
  checkError(Status, "amd_comgr_create_data_set");

  // Bitcode1
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC_BUNDLE, &DataBitcode1);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataBitcode1, SizeBitcode1, BufBitcode1);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataBitcode1, "square");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetBundled, DataBitcode1);
  checkError(Status, "amd_comgr_data_set_add");

  // Bitcode2
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC_BUNDLE, &DataBitcode2);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataBitcode2, SizeBitcode2, BufBitcode2);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataBitcode2, ""); // test blank name
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetBundled, DataBitcode2);
  checkError(Status, "amd_comgr_data_set_add");

  // Archive
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_AR_BUNDLE, &DataArchive);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataArchive, SizeArchive, BufArchive);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataArchive, "cube");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetBundled, DataArchive);
  checkError(Status, "amd_comgr_data_set_add");

  // Set up ActionInfo
  Status = amd_comgr_create_action_info(&ActionInfo);
  checkError(Status, "amd_comgr_create_action_info");

  Status =
      amd_comgr_action_info_set_language(ActionInfo, AMD_COMGR_LANGUAGE_HIP);
  checkError(Status, "amd_comgr_action_info_set_language");

  Status = amd_comgr_action_info_set_isa_name(ActionInfo, IsaName);

  // Unbundle
  Status = amd_comgr_create_data_set(&DataSetLinked);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, ActionInfo,
                               DataSetBundled, DataSetLinked);
  checkError(Status, "amd_comgr_do_action");

  // Check Linked bitcode count
  size_t Count;
  Status = amd_comgr_action_data_count(DataSetLinked,
                                       AMD_COMGR_DATA_KIND_BC, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("Bundled bitcode linking: "
           "produced %zu bitcodes (expected 1)\n",
           Count);
    exit(1);
  }

  // Compile to relocatable
  Status = amd_comgr_create_data_set(&DataSetReloc);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                               ActionInfo, DataSetLinked, DataSetReloc);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_count(DataSetReloc,
                                       AMD_COMGR_DATA_KIND_RELOCATABLE, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE Failed: "
           "produced %zu source objects (expected 1)\n",
           Count);
    exit(1);
  }

  // Compile to executable
  Status = amd_comgr_create_data_set(&DataSetExec);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_action_info_set_option_list(ActionInfo, NULL, 0);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               ActionInfo, DataSetReloc, DataSetExec);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_count(DataSetExec,
                                       AMD_COMGR_DATA_KIND_EXECUTABLE, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE Failed: "
           "produced %zu executable objects (expected 1)\n",
           Count);
    exit(1);
  }

  // Cleanup
  Status = amd_comgr_destroy_action_info(ActionInfo);
  checkError(Status, "amd_comgr_destroy_action_info");
  Status = amd_comgr_destroy_data_set(DataSetBundled);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetLinked);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetReloc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetExec);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_release_data(DataBitcode1);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataBitcode2);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataArchive);
  checkError(Status, "amd_comgr_release_data");

  free(BufBitcode1);
  free(BufBitcode2);
  free(BufArchive);
}
