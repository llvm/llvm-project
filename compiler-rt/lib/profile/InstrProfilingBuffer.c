/*===- InstrProfilingBuffer.c - Write instrumentation to a memory buffer --===*\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include "InstrProfilingPort.h"

/* When continuous mode is enabled (%c), this parameter is set to 1.
 *
 * This parameter is defined here in InstrProfilingBuffer.o, instead of in
 * InstrProfilingFile.o, to sequester all libc-dependent code in
 * InstrProfilingFile.o. The test `instrprof-without-libc` will break if this
 * layering is violated. */
static int ContinuouslySyncProfile = 0;

COMPILER_RT_VISIBILITY int __llvm_profile_is_continuous_mode_enabled(void) {
  return ContinuouslySyncProfile;
}

COMPILER_RT_VISIBILITY void __llvm_profile_enable_continuous_mode(void) {
  ContinuouslySyncProfile = 1;
}

COMPILER_RT_VISIBILITY
uint64_t __llvm_profile_get_size_for_buffer(void) {
  const __llvm_profile_data *DataBegin = __llvm_profile_begin_data();
  const __llvm_profile_data *DataEnd = __llvm_profile_end_data();
  const uint64_t *CountersBegin = __llvm_profile_begin_counters();
  const uint64_t *CountersEnd = __llvm_profile_end_counters();
  const char *NamesBegin = __llvm_profile_begin_names();
  const char *NamesEnd = __llvm_profile_end_names();

  return __llvm_profile_get_size_for_buffer_internal(
      DataBegin, DataEnd, CountersBegin, CountersEnd, NamesBegin, NamesEnd);
}

COMPILER_RT_VISIBILITY
uint64_t __llvm_profile_get_data_size(const __llvm_profile_data *Begin,
                                      const __llvm_profile_data *End) {
  intptr_t BeginI = (intptr_t)Begin, EndI = (intptr_t)End;
  return ((EndI + sizeof(__llvm_profile_data) - 1) - BeginI) /
         sizeof(__llvm_profile_data);
}

/// Calculate the number of padding bytes needed to add to \p Offset in order
/// for (\p Offset + Padding) to be page-aligned.
static uint64_t calculateBytesNeededToPageAlign(uint64_t Offset,
                                                unsigned PageSize) {
  uint64_t OffsetModPage = Offset % PageSize;
  if (OffsetModPage > 0)
    return PageSize - OffsetModPage;
  return 0;
}

COMPILER_RT_VISIBILITY
void __llvm_profile_get_padding_sizes_for_counters(
    uint64_t DataSize, uint64_t CountersSize, uint64_t NamesSize,
    uint64_t *PaddingBytesBeforeCounters, uint64_t *PaddingBytesAfterCounters,
    uint64_t *PaddingBytesAfterNames) {
  if (!__llvm_profile_is_continuous_mode_enabled()) {
    *PaddingBytesBeforeCounters = 0;
    *PaddingBytesAfterCounters = 0;
    *PaddingBytesAfterNames = __llvm_profile_get_num_padding_bytes(NamesSize);
    return;
  }

  // In continuous mode, the file offsets for headers and for the start of
  // counter sections need to be page-aligned.
  unsigned PageSize = getpagesize();
  uint64_t DataSizeInBytes = DataSize * sizeof(__llvm_profile_data);
  uint64_t CountersSizeInBytes = CountersSize * sizeof(uint64_t);
  *PaddingBytesBeforeCounters = calculateBytesNeededToPageAlign(
      sizeof(__llvm_profile_header) + DataSizeInBytes, PageSize);
  *PaddingBytesAfterCounters =
      calculateBytesNeededToPageAlign(CountersSizeInBytes, PageSize);
  *PaddingBytesAfterNames =
      calculateBytesNeededToPageAlign(NamesSize, PageSize);
}

COMPILER_RT_VISIBILITY
uint64_t __llvm_profile_get_size_for_buffer_internal(
    const __llvm_profile_data *DataBegin, const __llvm_profile_data *DataEnd,
    const uint64_t *CountersBegin, const uint64_t *CountersEnd,
    const char *NamesBegin, const char *NamesEnd) {
  /* Match logic in __llvm_profile_write_buffer(). */
  const uint64_t NamesSize = (NamesEnd - NamesBegin) * sizeof(char);
  uint64_t DataSize = __llvm_profile_get_data_size(DataBegin, DataEnd);
  uint64_t CountersSize = CountersEnd - CountersBegin;

  /* Determine how much padding is needed before/after the counters and after
   * the names. */
  uint64_t PaddingBytesBeforeCounters, PaddingBytesAfterCounters,
      PaddingBytesAfterNames;
  __llvm_profile_get_padding_sizes_for_counters(
      DataSize, CountersSize, NamesSize, &PaddingBytesBeforeCounters,
      &PaddingBytesAfterCounters, &PaddingBytesAfterNames);

  return sizeof(__llvm_profile_header) +
         (DataSize * sizeof(__llvm_profile_data)) + PaddingBytesBeforeCounters +
         (CountersSize * sizeof(uint64_t)) + PaddingBytesAfterCounters +
         NamesSize + PaddingBytesAfterNames;
}

COMPILER_RT_VISIBILITY
void initBufferWriter(ProfDataWriter *BufferWriter, char *Buffer) {
  BufferWriter->Write = lprofBufferWriter;
  BufferWriter->WriterCtx = Buffer;
}

COMPILER_RT_VISIBILITY int __llvm_profile_write_buffer(char *Buffer) {
  ProfDataWriter BufferWriter;
  initBufferWriter(&BufferWriter, Buffer);
  return lprofWriteData(&BufferWriter, 0, 0);
}

COMPILER_RT_VISIBILITY int __llvm_profile_write_buffer_internal(
    char *Buffer, const __llvm_profile_data *DataBegin,
    const __llvm_profile_data *DataEnd, const uint64_t *CountersBegin,
    const uint64_t *CountersEnd, const char *NamesBegin, const char *NamesEnd) {
  ProfDataWriter BufferWriter;
  initBufferWriter(&BufferWriter, Buffer);
  return lprofWriteDataImpl(&BufferWriter, DataBegin, DataEnd, CountersBegin,
                            CountersEnd, 0, NamesBegin, NamesEnd, 0);
}
