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

/// -------
//  Manual recreation of Comgr bundle linking
//
//    // Create bitcode bundles
//    clang -c --offload-arch=gfx900 -emit-llvm -fgpu-rdc \
//    --gpu-bundle-output square.hip cube.hip
//
//    // Create object file bundles
//    clang -c --offload-arch=gfx900 --gpu-bundle-output \
//    double.hip
//
//    // Create archive bundle
//    llvm-ar rc cube.a cube.bc
//
//    // Manually unbundle bitcode bundle
//    clang-offload-bundler -type=bc \
//    -targets=hip-amdgcn-amd-amdhsa-gfx900 \
//    -input=square.bc -output=square-gfx900.bc \
//    -unbundle -allow-missing-bundles
//
//    // Manually unbundle object file bundle
//    clang-offload-bundler -type=o \
//    -targets=hip-amdgcn-amd-amdhsa-gfx900 \
//    -input=double.o -output=double-gfx900.o \
//    -unbundle -allow-missing-bundles
//
//    // Manually unbundle archive bundle
//    clang-offload-bundler -type=a \
//    -targets=hip-amdgcn-amd-amdhsa-gfx900 \
//    -input=cube.a -output=cube-gfx900.a \
//    -unbundle -allow-missing-bundles \
//    -hip-openmp-compatible

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int Argc, char *Argv[]) {
  char *BufBitcode, *BufObjectFile, *BufArchive;
  size_t SizeBitcode, SizeObjectFile, SizeArchive;
  amd_comgr_data_t DataBitcode, DataObjectFile, DataArchive;
  amd_comgr_data_set_t DataSetBundled, DataSetUnbundled,
                       DataSetLinked, DataSetReloc,
                       DataSetExec;
  amd_comgr_action_info_t ActionInfoUnbundle, ActionInfoLink;
  amd_comgr_status_t Status;

  SizeBitcode = setBuf("./source/square.bc", &BufBitcode);
  SizeObjectFile = setBuf("./source/double.o", &BufObjectFile);
  SizeArchive = setBuf("./source/cube.a", &BufArchive);

  // Create Bundled dataset
  Status = amd_comgr_create_data_set(&DataSetBundled);
  checkError(Status, "amd_comgr_create_data_set");

  // Bitcode
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC_BUNDLE, &DataBitcode);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataBitcode, SizeBitcode, BufBitcode);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataBitcode, "square");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetBundled, DataBitcode);
  checkError(Status, "amd_comgr_data_set_add");

  // ObjectFile
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_OBJ_BUNDLE,
                                 &DataObjectFile);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataObjectFile, SizeObjectFile, BufObjectFile);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataObjectFile, ""); // test blank name
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetBundled, DataObjectFile);
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

  // Unbundle explicitly via UNBUNDLE action
  {
    // Set up ActionInfo
    Status = amd_comgr_create_action_info(&ActionInfoUnbundle);
    checkError(Status, "amd_comgr_create_action_info");

    Status =
      amd_comgr_action_info_set_language(ActionInfoUnbundle,
                                         AMD_COMGR_LANGUAGE_HIP);
    checkError(Status, "amd_comgr_action_info_set_language");

    const char *BundleEntryIDs[] = {"host-x86_64-unknown-linux-gnu",
      "hip-amdgcn-amd-amdhsa-gfx900"};
    size_t BundleEntryIDsCount =
      sizeof(BundleEntryIDs) / sizeof(BundleEntryIDs[0]);
    Status = amd_comgr_action_info_set_bundle_entry_ids(ActionInfoUnbundle,
                                                        BundleEntryIDs, 2);

    // Unbundle
    Status = amd_comgr_create_data_set(&DataSetUnbundled);
    checkError(Status, "amd_comgr_create_data_set");
    Status = amd_comgr_do_action(AMD_COMGR_ACTION_UNBUNDLE, ActionInfoUnbundle,
                                 DataSetBundled, DataSetUnbundled);
    checkError(Status, "amd_comgr_do_action");

    // Check Bitcode count
    size_t Count;
    Status = amd_comgr_action_data_count(DataSetUnbundled,
                                         AMD_COMGR_DATA_KIND_BC, &Count);
    checkError(Status, "amd_comgr_action_data_count");

    if (Count != 2) {
      printf("Unbundling: "
             "produced %zu bitcodes (expected 2)\n",
             Count);
      exit(1);
    }

    // Check ObjectFile count
    Status = amd_comgr_action_data_count(DataSetUnbundled,
                                         AMD_COMGR_DATA_KIND_EXECUTABLE,
                                         &Count);
    checkError(Status, "amd_comgr_action_data_count");

    if (Count != 2) {
      printf("Unbundling: "
             "produced %zu object files (expected 2)\n",
             Count);
      exit(1);
    }

    // Check Archive count
    Status = amd_comgr_action_data_count(DataSetUnbundled,
                                         AMD_COMGR_DATA_KIND_AR, &Count);
    checkError(Status, "amd_comgr_action_data_count");

    if (Count != 2) {
      printf("Unbundle: "
             "produced %zu archives (expected 2)\n",
             Count);
      exit(1);
    }

    // Check Bundle Entry IDs
    size_t BundleCount;
    Status = amd_comgr_action_info_get_bundle_entry_id_count(ActionInfoUnbundle,
                                                        &BundleCount);
    checkError(Status, "amd_comgr_action_info_get_bundle_entry_id_count");

    for (int i = 0; i < BundleCount; i++) {

      size_t Size;
      Status = amd_comgr_action_info_get_bundle_entry_id(ActionInfoUnbundle, i,
                                                         &Size, NULL);
      checkError(Status, "amd_comgr_action_info_get_bundle_entry_id");

      char *BundleID = calloc(Size, sizeof(char));
      Status = amd_comgr_action_info_get_bundle_entry_id(ActionInfoUnbundle, i,
                                                         &Size, BundleID);
      checkError(Status, "amd_comgr_action_info_get_bundle_entry_id");

      if (strcmp(BundleID, BundleEntryIDs[i])) {
        printf("BundleEntryID mismatch. Expected \"%s\", returned \"%s\"\n",
               BundleEntryIDs[i],
               BundleID);
        checkError(AMD_COMGR_STATUS_ERROR,
                   "amd_comgr_action_info_get_bundle_entry_id");
      }

      free(BundleID);
    }
  }

  // Unbundle silently via LINK action
  {
    // Set up ActionInfo
    Status = amd_comgr_create_action_info(&ActionInfoLink);
    checkError(Status, "amd_comgr_create_action_info");

    Status =
      amd_comgr_action_info_set_language(ActionInfoLink,
                                         AMD_COMGR_LANGUAGE_HIP);
    checkError(Status, "amd_comgr_action_info_set_language");

    const char *IsaName = "amdgcn-amd-amdhsa--gfx900";
    Status = amd_comgr_action_info_set_isa_name(ActionInfoLink, IsaName);

    // Unbundle
    Status = amd_comgr_create_data_set(&DataSetLinked);
    checkError(Status, "amd_comgr_create_data_set");
    Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, ActionInfoLink,
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
                                 ActionInfoLink, DataSetLinked, DataSetReloc);
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

    Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                                 ActionInfoLink, DataSetReloc, DataSetExec);
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
  }

  // Cleanup
  Status = amd_comgr_destroy_action_info(ActionInfoUnbundle);
  checkError(Status, "amd_comgr_destroy_action_info");
  Status = amd_comgr_destroy_action_info(ActionInfoLink);
  checkError(Status, "amd_comgr_destroy_action_info");
  Status = amd_comgr_destroy_data_set(DataSetBundled);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetUnbundled);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetLinked);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetReloc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetExec);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_release_data(DataBitcode);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataObjectFile);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataArchive);
  checkError(Status, "amd_comgr_release_data");

  free(BufBitcode);
  free(BufObjectFile);
  free(BufArchive);

  return 0;
}
