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

#include "amd_comgr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

int main(int argc, char *argv[]) {
  long size;
  char *buf;
  amd_comgr_data_t dataObject;
  amd_comgr_status_t status;
  int count = 1;

  size = setBuf(TEST_OBJ_DIR "/shared.so", &buf);

  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &dataObject);
  checkError(status, "amd_comgr_create_data");

  status = amd_comgr_set_data(dataObject, size, buf);
  checkError(status, "amd_comgr_set_data");

  status = amd_comgr_iterate_symbols(dataObject, print_symbol, &count);
  checkError(status, "amd_comgr_iterate_symbols");

  status = amd_comgr_release_data(dataObject);
  checkError(status, "amd_comgr_release_data");
  free(buf);

  return 0;
}
