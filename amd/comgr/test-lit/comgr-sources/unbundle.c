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

int main(int argc, char *argv[]) {
  char *BundleData;
  size_t BundleSize;

  if (argc < 4) {
    printf("Usage: %s <bc bundle> <arch> <bc output>\n", argv[0]);
    return -1;
  }

  const char *BundlePath = argv[1];
  const char *Arch = argv[2];
  const char *BitcodePath = argv[3];

  amd_comgr_data_t OneBundle;
  amd_comgr_data_set_t InputBundles;

  BundleSize = setBuf(BundlePath, &BundleData);

  amd_comgr_(create_data_set(&InputBundles));
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_BC_BUNDLE, &OneBundle));
  amd_comgr_(set_data(OneBundle, BundleSize, BundleData));
  amd_comgr_(set_data_name(OneBundle, "bundle.bc"));
  amd_comgr_(data_set_add(InputBundles, OneBundle));

  amd_comgr_data_set_t OutputBitcode;
  amd_comgr_(create_data_set(&OutputBitcode));

  amd_comgr_action_info_t DataAction;
  amd_comgr_(create_action_info(&DataAction));

  const char *AllArch[] = {Arch};
  amd_comgr_(action_info_set_bundle_entry_ids(DataAction, AllArch, 1));
  amd_comgr_(do_action(AMD_COMGR_ACTION_UNBUNDLE, DataAction, InputBundles,
                       OutputBitcode));

  size_t Count;
  amd_comgr_(action_data_count(OutputBitcode, AMD_COMGR_DATA_KIND_BC, &Count));

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC Failed: "
           "produced %zu BC objects (expected 1)\n",
           Count);
    exit(1);
  }

  amd_comgr_data_t OneBitcode;
  amd_comgr_(action_data_get_data(OutputBitcode, AMD_COMGR_DATA_KIND_BC, 0,
                                  &OneBitcode));

  size_t BufferSize;
  amd_comgr_(get_data(OneBitcode, &BufferSize, 0x0));
  char *Buffer = (char *)malloc(BufferSize);
  amd_comgr_(get_data(OneBitcode, &BufferSize, Buffer));

  FILE *BitcodeFile = fopen(BitcodePath, "wb");
  fwrite(Buffer, 1, BufferSize, BitcodeFile);
  fclose(BitcodeFile);

  free(Buffer);
  amd_comgr_(release_data(OneBitcode));
  amd_comgr_(release_data(OneBundle));
  amd_comgr_(destroy_action_info(DataAction));
  amd_comgr_(destroy_data_set(OutputBitcode));
  amd_comgr_(destroy_data_set(InputBundles));

  return 0;
}
