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

#ifndef COMGR_TEST_COMMON_H
#define COMGR_TEST_COMMON_H

#include "amd_comgr.h"
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#else // Windows
#include <io.h>
#endif
#include <errno.h>
#include <fcntl.h>

void fail(const char *format, ...) {
  va_list ap;
  va_start(ap, format);

  printf("FAILED: ");
  vprintf(format, ap);
  printf("\n");

  va_end(ap);

  exit(1);
}

int setBuf(const char *infile, char **buf) {
  FILE *fp;
  long size;

  fp = fopen(infile, "rb");
  if (!fp)
    fail("fopen : %s", infile);
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

void checkError(amd_comgr_status_t status, const char *str) {
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    const char *statusStr;
    printf("FAILED: %s\n", str);
    status = amd_comgr_status_string(status, &statusStr);
    if (status == AMD_COMGR_STATUS_SUCCESS)
      printf(" REASON: %s\n", statusStr);
    exit(1);
  }
}

void dumpData(amd_comgr_data_t Data, const char *OutFile) {
  size_t size;
  char *bytes = NULL;
  amd_comgr_status_t status;

  status = amd_comgr_get_data(Data, &size, NULL);
  checkError(status, "amd_comgr_get_data");

  bytes = (char *)malloc(size);
  if (!bytes)
    fail("malloc");

  status = amd_comgr_get_data(Data, &size, bytes);
  checkError(status, "amd_comgr_get_data");

  FILE *fp = fopen(OutFile, "wb");
  if (!fp)
    fail("fopen : %s", OutFile);

  size_t ret = fwrite(bytes, sizeof(char), size, fp);
  if (ret != size)
    fail("fwrite");

  fclose(fp);
}

amd_comgr_status_t printSymbol(amd_comgr_symbol_t symbol, void *userData) {
  amd_comgr_status_t status;

  size_t nlen;
  status = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_NAME_LENGTH,
                                     (void *)&nlen);
  checkError(status, "amd_comgr_symbol_get_info_1");

  char *name = (char *)malloc(nlen + 1);
  status = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_NAME,
                                     (void *)name);
  checkError(status, "amd_comgr_symbol_get_info_2");

  amd_comgr_symbol_type_t type;
  status = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_TYPE,
                                     (void *)&type);
  checkError(status, "amd_comgr_symbol_get_info_3");

  uint64_t size;
  status = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_SIZE,
                                     (void *)&size);
  checkError(status, "amd_comgr_symbol_get_info_4");

  bool undefined;
  status = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED,
                                     (void *)&undefined);
  checkError(status, "amd_comgr_symbol_get_info_5");

  uint64_t value;
  status = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_VALUE,
                                     (void *)&value);
  checkError(status, "amd_comgr_symbol_get_info_6");

  printf("%d:  name=%s, type=%d, size=%" PRIu64 ", undef:%d, value:%" PRIu64
         "I64u\n",
         *(int *)userData, name, type, size, undefined ? 1 : 0, value);
  *(int *)userData += 1;

  free(name);

  return status;
}

amd_comgr_status_t printEntry(amd_comgr_metadata_node_t key,
                              amd_comgr_metadata_node_t value, void *data) {
  amd_comgr_metadata_kind_t kind;
  amd_comgr_metadata_node_t son;
  amd_comgr_status_t status;
  size_t size;
  char *keybuf;
  char *valbuf;
  int *indent = (int *)data;

  // assume key to be string in this test function
  status = amd_comgr_get_metadata_kind(key, &kind);
  checkError(status, "amd_comgr_get_metadata_kind");
  if (kind != AMD_COMGR_METADATA_KIND_STRING)
    return AMD_COMGR_STATUS_ERROR;
  status = amd_comgr_get_metadata_string(key, &size, NULL);
  checkError(status, "amd_comgr_get_metadata_string");
  keybuf = (char *)calloc(size, sizeof(char));
  if (!keybuf)
    fail("calloc");
  status = amd_comgr_get_metadata_string(key, &size, keybuf);
  checkError(status, "amd_comgr_get_metadata_string");

  status = amd_comgr_get_metadata_kind(value, &kind);
  checkError(status, "amd_comgr_get_metadata_kind");
  for (int i = 0; i < *indent; i++)
    printf("  ");

  switch (kind) {
  case AMD_COMGR_METADATA_KIND_STRING: {
    printf("%s  :  ", size ? keybuf : "");
    status = amd_comgr_get_metadata_string(value, &size, NULL);
    checkError(status, "amd_comgr_get_metadata_string");
    valbuf = (char *)calloc(size, sizeof(char));
    if (!valbuf)
      fail("calloc");
    status = amd_comgr_get_metadata_string(value, &size, valbuf);
    checkError(status, "amd_comgr_get_metadata_string");
    printf(" %s\n", valbuf);
    free(valbuf);
    break;
  }
  case AMD_COMGR_METADATA_KIND_LIST: {
    *indent += 1;
    status = amd_comgr_get_metadata_list_size(value, &size);
    checkError(status, "amd_comgr_get_metadata_list_size");
    printf("LIST %s %zd entries = \n", keybuf, size);
    for (size_t i = 0; i < size; i++) {
      status = amd_comgr_index_list_metadata(value, i, &son);
      checkError(status, "amd_comgr_index_list_metadata");
      status = printEntry(key, son, data);
      checkError(status, "printEntry");
      status = amd_comgr_destroy_metadata(son);
      checkError(status, "amd_comgr_destroy_metadata");
    }
    *indent = *indent > 0 ? *indent - 1 : 0;
    break;
  }
  case AMD_COMGR_METADATA_KIND_MAP: {
    *indent += 1;
    status = amd_comgr_get_metadata_map_size(value, &size);
    checkError(status, "amd_comgr_get_metadata_map_size");
    printf("MAP %zd entries = \n", size);
    status = amd_comgr_iterate_map_metadata(value, printEntry, data);
    checkError(status, "amd_comgr_iterate_map_metadata");
    *indent = *indent > 0 ? *indent - 1 : 0;
    break;
  }
  default:
    free(keybuf);
    return AMD_COMGR_STATUS_ERROR;
  } // switch

  free(keybuf);
  return AMD_COMGR_STATUS_SUCCESS;
}

void checkLogs(const char *id, amd_comgr_data_set_t dataSet,
               const char *expected) {
  amd_comgr_status_t status;

  size_t count;
  status =
      amd_comgr_action_data_count(dataSet, AMD_COMGR_DATA_KIND_LOG, &count);
  checkError(status, "amd_comgr_action_data_count");

  for (size_t i = 0; i < count; i++) {
    amd_comgr_data_t data;
    status = amd_comgr_action_data_get_data(dataSet, AMD_COMGR_DATA_KIND_LOG, i,
                                            &data);
    checkError(status, "amd_comgr_action_data_get_data");

    size_t size;
    status = amd_comgr_get_data(data, &size, NULL);
    checkError(status, "amd_comgr_get_data");

    char *bytes = (char *)malloc(size + 1);
    if (!bytes)
      fail("malloc");
    status = amd_comgr_get_data(data, &size, bytes);
    checkError(status, "amd_comgr_get_data");
    bytes[size] = '\0';

    if (!strstr(bytes, expected)) {
      printf("%s failed: expected substring \"%s\" not found in log:\n%s", id,
             expected, bytes);
      exit(1);
    }

    free(bytes);

    status = amd_comgr_release_data(data);
    checkError(status, "amd_comgr_release_data");
  }
}

// FIXME: This should probably be defined by Comgr
const char *dataKindString(amd_comgr_data_kind_t dataKind) {
  static const char *strings[AMD_COMGR_DATA_KIND_FATBIN + 1] = {
      [AMD_COMGR_DATA_KIND_UNDEF] = "AMD_COMGR_DATA_KIND_UNDEF",
      [AMD_COMGR_DATA_KIND_SOURCE] = "AMD_COMGR_DATA_KIND_SOURCE",
      [AMD_COMGR_DATA_KIND_INCLUDE] = "AMD_COMGR_DATA_KIND_INCLUDE",
      [AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER] =
          "AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER",
      [AMD_COMGR_DATA_KIND_DIAGNOSTIC] = "AMD_COMGR_DATA_KIND_DIAGNOSTIC",
      [AMD_COMGR_DATA_KIND_LOG] = "AMD_COMGR_DATA_KIND_LOG",
      [AMD_COMGR_DATA_KIND_BC] = "AMD_COMGR_DATA_KIND_BC",
      [AMD_COMGR_DATA_KIND_RELOCATABLE] = "AMD_COMGR_DATA_KIND_RELOCATABLE",
      [AMD_COMGR_DATA_KIND_EXECUTABLE] = "AMD_COMGR_DATA_KIND_EXECUTABLE",
      [AMD_COMGR_DATA_KIND_BYTES] = "AMD_COMGR_DATA_KIND_BYTES",
      [AMD_COMGR_DATA_KIND_FATBIN] = "AMD_COMGR_DATA_KIND_FATBIN",
  };
  return strings[dataKind];
}

void checkCount(const char *id, amd_comgr_data_set_t dataSet,
                amd_comgr_data_kind_t dataKind, size_t expected) {
  amd_comgr_status_t status;

  size_t count;
  status = amd_comgr_action_data_count(dataSet, dataKind, &count);
  checkError(status, "checkCount:amd_comgr_action_data_count");

  if (count != expected)
    fail("%s failed: produced %zu %s objects (expected %zu)\n", id, count,
         dataKindString(dataKind), expected);
}

size_t WriteFile(int FD, const char *Buffer, size_t Size) {
  size_t BytesWritten = 0;

  while (BytesWritten < Size) {
#if defined(_WIN32) || defined(_WIN64)
    size_t Ret =
        _write(FD, Buffer + BytesWritten, (unsigned int)(Size - BytesWritten));
#else
    size_t Ret = write(FD, Buffer + BytesWritten, Size - BytesWritten);
#endif
    if (Ret == 0) {
      break;
    } else if (Ret < 0) {
      if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
        break;
      }
      printf("Write failed with errno %d\n", errno);
    } else {
      BytesWritten += Ret;
    }
  }

  return BytesWritten;
}

#endif // COMGR_TEST_COMMON_H
