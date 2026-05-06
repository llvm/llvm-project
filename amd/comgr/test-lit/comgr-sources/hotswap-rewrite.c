//===- hotswap-rewrite.c - Test HotSwap rewrite API ----------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Canonical hotswap input/output driver for lit tests. Loads an ELF, runs
/// amd_comgr_hotswap_rewrite, and optionally dumps the output and/or checks
/// that a second rewrite produces identical output (idempotency).
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    amd_comgr_data_t dummy_output;
    amd_comgr_data_t dummy_input = {0};
    amd_comgr_status_t Status =
        amd_comgr_hotswap_rewrite(dummy_input, NULL, NULL, &dummy_output);
    if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
      fail("rewrite with NULL args: expected INVALID_ARGUMENT");
    printf("NULL_ARGS: INVALID_ARGUMENT\n");
    return 0;
  }

  if (argc < 4)
    fail("usage: hotswap-rewrite <elf_file> <source_isa> <target_isa> "
         "[--zero-size] [--output <path>] [--dump <file>] "
         "[--check-idempotent]");

  const char *ElfFile = argv[1];
  const char *SourceISA = argv[2];
  const char *TargetISA = argv[3];
  int ZeroSize = 0;
  const char *OutputPath = NULL;
  const char *DumpFile = NULL;
  int CheckIdempotent = 0;

  for (int I = 4; I < argc; ++I) {
    if (strcmp(argv[I], "--zero-size") == 0)
      ZeroSize = 1;
    else if (strcmp(argv[I], "--output") == 0 && I + 1 < argc)
      OutputPath = argv[++I];
    else if (strcmp(argv[I], "--dump") == 0 && I + 1 < argc)
      DumpFile = argv[++I];
    else if (strcmp(argv[I], "--check-idempotent") == 0)
      CheckIdempotent = 1;
    else {
      fprintf(stderr, "error: unknown argument: %s\n", argv[I]);
      return 1;
    }
  }

  char *ElfBuf;
  size_t ElfSize = (size_t)setBuf(ElfFile, &ElfBuf);

  amd_comgr_data_t InputData;
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &InputData));
  if (!ZeroSize) {
    amd_comgr_(set_data(InputData, ElfSize, ElfBuf));
  }

  amd_comgr_data_t OutputData;
  amd_comgr_status_t Status =
      amd_comgr_hotswap_rewrite(InputData, SourceISA, TargetISA, &OutputData);

  if (Status == AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT) {
    printf("RESULT: INVALID_ARGUMENT\n");
    amd_comgr_(release_data(InputData));
    free(ElfBuf);
    return 0;
  }

  if (Status != AMD_COMGR_STATUS_SUCCESS)
    fail("unexpected error status %d", (int)Status);

  size_t OutSize = 0;
  amd_comgr_(get_data(OutputData, &OutSize, NULL));

  if (OutputPath) {
    dumpData(OutputData, OutputPath);
    printf("RESULT: SUCCESS\n");
  } else if (DumpFile || CheckIdempotent) {
    printf("REWRITE: SUCCESS\n");

    if (DumpFile)
      dumpData(OutputData, DumpFile);

    if (CheckIdempotent) {
      amd_comgr_data_t Output2Data;
      Status = amd_comgr_hotswap_rewrite(OutputData, SourceISA, TargetISA,
                                         &Output2Data);
      if (Status != AMD_COMGR_STATUS_SUCCESS)
        fail("idempotent rewrite failed with status %d", (int)Status);

      size_t Output2Size;
      amd_comgr_(get_data(Output2Data, &Output2Size, NULL));

      char *Out1Buf = (char *)malloc(OutSize);
      if (!Out1Buf)
        fail("malloc failed");
      amd_comgr_(get_data(OutputData, &OutSize, Out1Buf));

      char *Out2Buf = (char *)malloc(Output2Size);
      if (!Out2Buf)
        fail("malloc failed");
      amd_comgr_(get_data(Output2Data, &Output2Size, Out2Buf));

      if (Output2Size == OutSize && memcmp(Out1Buf, Out2Buf, OutSize) == 0)
        printf("IDEMPOTENT: YES\n");
      else
        printf("IDEMPOTENT: NO (%zu vs %zu)\n", Output2Size, OutSize);

      free(Out1Buf);
      free(Out2Buf);
      amd_comgr_(release_data(Output2Data));
    }
  } else {
    if (OutSize != ElfSize)
      fail("output size %zu != input size %zu", OutSize, ElfSize);

    char *OutBuf = (char *)malloc(OutSize);
    if (!OutBuf)
      fail("malloc failed");
    amd_comgr_(get_data(OutputData, &OutSize, OutBuf));

    if (memcmp(OutBuf, ElfBuf, ElfSize) != 0)
      fail("output content differs from input");

    free(OutBuf);
    printf("RESULT: SUCCESS\n");
  }

  amd_comgr_(release_data(OutputData));
  amd_comgr_(release_data(InputData));
  free(ElfBuf);

  return 0;
}
