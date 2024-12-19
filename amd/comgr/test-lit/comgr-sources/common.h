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
#include <sys/syscall.h>
#include <unistd.h>
#else // Windows
#include <io.h>
#endif
#include <errno.h>
#include <fcntl.h>

#define amd_comgr_(call)                                \
  do {                                                  \
    amd_comgr_status_t status = amd_comgr_ ## call;     \
    if (status != AMD_COMGR_STATUS_SUCCESS) {           \
      const char* reason = "";                          \
      amd_comgr_status_string(status, &reason);         \
      fail(#call " failed: %s\n  file, line: %s, %d\n", \
           reason, __FILE__, __LINE__);                 \
    }                                                   \
  } while (false)

static void fail(const char *format, ...) {
  va_list ap;
  va_start(ap, format);

  printf("FAILED: ");
  vprintf(format, ap);
  printf("\n");

  va_end(ap);

  exit(1);
}

static int setBuf(const char *infile, char **buf) {
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

  *buf = (char *) malloc(size + 1);
  if (!*buf)
    fail("malloc");
  if (fread(*buf, size, 1, fp) != 1)
    fail("fread");
  if (fclose(fp) != 0)
    fail("fclose");
  (*buf)[size] = 0; // terminating zero
  return size;
}

static void dumpData(amd_comgr_data_t Data, const char *OutFile) {
  size_t size;
  char *bytes = NULL;

  amd_comgr_(get_data(Data, &size, NULL));

  bytes = (char *)malloc(size);
  if (!bytes)
    fail("malloc");

  amd_comgr_(get_data(Data, &size, bytes));

  FILE *fp = fopen(OutFile, "wb");
  if (!fp)
    fail("fopen : %s", OutFile);

  size_t ret = fwrite(bytes, sizeof(char), size, fp);
  if (ret != size)
    fail("fwrite");

  free(bytes);
  fclose(fp);
}

#endif // COMGR_TEST_COMMON_H
