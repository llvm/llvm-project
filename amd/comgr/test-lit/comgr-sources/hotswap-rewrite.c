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
/// the hotswap rewrite API, and optionally dumps the output and/or reruns the
/// same request and compares the two outputs.
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

static amd_comgr_status_t
runRewrite(amd_comgr_data_t InputData, const char *SourceISA,
           const char *TargetISA, uint64_t RewriteFlags, int BadOptionsSize,
           int BadOptionsFlags, amd_comgr_data_t *OutputData) {
  if (RewriteFlags == AMD_COMGR_HOTSWAP_REWRITE_FLAG_NONE && !BadOptionsSize &&
      !BadOptionsFlags)
    return amd_comgr_hotswap_rewrite(InputData, SourceISA, TargetISA,
                                     OutputData);

  amd_comgr_hotswap_rewrite_options_t Options = {
      sizeof(amd_comgr_hotswap_rewrite_options_t), RewriteFlags};
  if (BadOptionsSize)
    Options.size = 0;
  if (BadOptionsFlags)
    Options.flags = 0x4;

  return amd_comgr_hotswap_rewrite_with_options(InputData, SourceISA, TargetISA,
                                                &Options, OutputData);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    amd_comgr_data_t dummy_output;
    amd_comgr_data_t dummy_input = {0};
    amd_comgr_status_t Status =
        amd_comgr_hotswap_rewrite(dummy_input, NULL, NULL, &dummy_output);
    if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
      fail("rewrite with NULL args: expected INVALID_ARGUMENT");

    Status = amd_comgr_hotswap_rewrite_with_options(dummy_input, NULL, NULL,
                                                    NULL, &dummy_output);
    if (Status != AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
      fail("rewrite with NULL options: expected INVALID_ARGUMENT");

    printf("NULL_ARGS: INVALID_ARGUMENT\n");
    return 0;
  }

  if (argc < 4)
    fail(
        "usage: hotswap-rewrite <elf_file> <source_isa> <target_isa> "
        "[--entry-trampolines] [--strict-mode] [--bad-options-size] "
        "[--bad-options-flags] [--zero-size] [--output <path>] [--dump <file>] "
        "[--check-idempotent] [--expect-status <status>]");

  const char *ElfFile = argv[1];
  const char *SourceISA = argv[2];
  const char *TargetISA = argv[3];
  int ZeroSize = 0;
  const char *OutputPath = NULL;
  const char *DumpFile = NULL;
  const char *ExpectStatus = NULL;
  int CheckIdempotent = 0;
  int BadOptionsSize = 0;
  int BadOptionsFlags = 0;
  uint64_t RewriteFlags = AMD_COMGR_HOTSWAP_REWRITE_FLAG_NONE;

  for (int I = 4; I < argc; ++I) {
    if (strcmp(argv[I], "--zero-size") == 0)
      ZeroSize = 1;
    else if (strcmp(argv[I], "--entry-trampolines") == 0)
      RewriteFlags |= AMD_COMGR_HOTSWAP_REWRITE_FLAG_ENTRY_TRAMPOLINES;
    else if (strcmp(argv[I], "--strict-mode") == 0)
      RewriteFlags |= AMD_COMGR_HOTSWAP_REWRITE_FLAG_STRICT_MODE;
    else if (strcmp(argv[I], "--bad-options-size") == 0)
      BadOptionsSize = 1;
    else if (strcmp(argv[I], "--bad-options-flags") == 0)
      BadOptionsFlags = 1;
    else if (strcmp(argv[I], "--output") == 0 && I + 1 < argc)
      OutputPath = argv[++I];
    else if (strcmp(argv[I], "--dump") == 0 && I + 1 < argc)
      DumpFile = argv[++I];
    else if (strcmp(argv[I], "--check-idempotent") == 0)
      CheckIdempotent = 1;
    else if (strcmp(argv[I], "--expect-status") == 0 && I + 1 < argc)
      ExpectStatus = argv[++I];
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
      runRewrite(InputData, SourceISA, TargetISA, RewriteFlags, BadOptionsSize,
                 BadOptionsFlags, &OutputData);

  const char *StatusString;
  amd_comgr_(status_string(Status, &StatusString));

  if (ExpectStatus) {
    printf("RESULT: %s\n", StatusString);
    if (strcmp(StatusString, ExpectStatus) != 0)
      fail("expected status %s, saw %s", ExpectStatus, StatusString);
    if (Status == AMD_COMGR_STATUS_SUCCESS)
      amd_comgr_(release_data(OutputData));
    amd_comgr_(release_data(InputData));
    free(ElfBuf);
    return 0;
  }

  if (Status == AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT) {
    printf("RESULT: INVALID_ARGUMENT\n");
    amd_comgr_(release_data(InputData));
    free(ElfBuf);
    return 0;
  }

  if (Status != AMD_COMGR_STATUS_SUCCESS)
    fail("unexpected error status %s", StatusString);

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
      Status = runRewrite(OutputData, SourceISA, TargetISA, RewriteFlags,
                          BadOptionsSize, BadOptionsFlags, &Output2Data);
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
