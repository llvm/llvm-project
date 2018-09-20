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

#ifndef COMGR_TEST_COMMON_H
#define COMGR_TEST_COMMON_H

void fail(const char *msg) {
  printf("FAILED: %s", msg);
  exit(1);
}

int setBuf(char *infile, char **buf) {
  FILE *fp;
  long size;

  fp = fopen(infile, "rb");
  if (!fp)
    fail("fopen");
  if (fseek(fp, 0L, SEEK_END) != 0)
    fail("fopen");
  size = ftell(fp);
  if (size == -1)
    fail("ftell");
  if (fseek(fp, 0, SEEK_SET) != 0)
    fail("fseek");

  *buf = malloc(size + 1);
  if (!*buf)
    fail("malloc");
  if (fread(*buf, size, 1, fp) != 1)
    fail("fread");
  if (fclose(fp) != 0)
    fail("fclose");
  (*buf)[size] = 0; // terminating zero
  return size;
}

void checkError(amd_comgr_status_t status, char *str)
{
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    const char *status_str;
    printf("FAILED: %s\n", str);
    status = amd_comgr_status_string(status, &status_str);
    printf(" REASON: %s\n", status_str);
    exit(1);
  }
}

amd_comgr_status_t print_symbol(
  amd_comgr_symbol_t symbol,
  void *user_data)
{
  amd_comgr_status_t status;

  int nlen;
  status = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_NAME_LENGTH, (void*)&nlen);
  checkError(status, "amd_comgr_symbol_get_info_1");

  char *name = (char *)malloc(nlen+1);
  status = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_NAME, (void*)name);
  checkError(status, "amd_comgr_symbol_get_info_2");

  amd_comgr_symbol_type_t type;
  status = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_TYPE, (void*)&type);
  checkError(status, "amd_comgr_symbol_get_info_3");

  uint64_t size;
  status = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_SIZE, (void*)&size);
  checkError(status, "amd_comgr_symbol_get_info_4");

  bool undefined;
  status = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED, (void*)&undefined);
  checkError(status, "amd_comgr_symbol_get_info_5");

  uint64_t value;
  status = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_VALUE, (void*)&value);
  checkError(status, "amd_comgr_symbol_get_info_6");

  printf("%d:  name=%s, type=%d, size=%lu, undef:%d, value:%lu\n",
         *(int*)user_data, name, type, size, undefined? 1 : 0, value);
  *(int*)user_data += 1;

  free(name);

  return status;
}

#endif // COMGR_TEST_COMMON_H
