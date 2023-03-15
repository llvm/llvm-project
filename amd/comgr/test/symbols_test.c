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

void expectSymbol(const char *ObjectFilename, const char *SymbolName,
                  amd_comgr_symbol_type_t ExpectedType) {
  long Size;
  char *Buf;
  amd_comgr_data_t DataObject;
  amd_comgr_symbol_t Symbol;
  amd_comgr_status_t Status;

  Size = setBuf(ObjectFilename, &Buf);

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataObject);
  checkError(Status, "amd_comgr_create_data");

  Status = amd_comgr_set_data(DataObject, Size, Buf);
  checkError(Status, "amd_comgr_set_data");

  Status = amd_comgr_symbol_lookup(DataObject, SymbolName, &Symbol);
  checkError(Status, "amd_comgr_symbol_lookup");

  amd_comgr_symbol_type_t Type;
  Status = amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_TYPE,
                                     (void *)&Type);
  checkError(Status, "amd_comgr_symbol_get_info");

  if (Type != ExpectedType) {
    fail("unexpected symbol type for symbol %s: expected %d, saw %d\n",
         SymbolName, ExpectedType, Type);
  }

  Status = amd_comgr_release_data(DataObject);
  checkError(Status, "amd_comgr_release_data");
  free(Buf);
}

int main(int argc, char *argv[]) {
  expectSymbol(TEST_OBJ_DIR "/shared-v2.so",
               "bazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
               AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL);
  expectSymbol(TEST_OBJ_DIR "/shared-v3.so",
               "bazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
               AMD_COMGR_SYMBOL_TYPE_FUNC);
  expectSymbol(TEST_OBJ_DIR "/shared-v2.so", "foo", AMD_COMGR_SYMBOL_TYPE_OBJECT);
  expectSymbol(TEST_OBJ_DIR "/shared-v3.so", "foo",
               AMD_COMGR_SYMBOL_TYPE_OBJECT);
  return 0;
}
