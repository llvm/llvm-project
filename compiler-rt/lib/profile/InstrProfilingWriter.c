/*===- InstrProfilingWriter.c - Write instrumentation to a file or buffer -===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include <string.h>

#define INSTR_PROF_VALUE_PROF_DATA
#include "InstrProfData.inc"
void (*FreeHook)(void *) = NULL;
void* (*CallocHook)(size_t, size_t) = NULL;
uint32_t VPBufferSize = 0;

/* The buffer writer is reponsponsible in keeping writer state
 * across the call.
 */
COMPILER_RT_VISIBILITY uint32_t llvmBufferWriter(ProfDataIOVec *IOVecs,
                                                 uint32_t NumIOVecs,
                                                 void **WriterCtx) {
  uint32_t I;
  char **Buffer = (char **)WriterCtx;
  for (I = 0; I < NumIOVecs; I++) {
    size_t Length = IOVecs[I].ElmSize * IOVecs[I].NumElm;
    memcpy(*Buffer, IOVecs[I].Data, Length);
    *Buffer += Length;
  }
  return 0;
}

static void llvmInitBufferIO(ProfBufferIO *BufferIO, WriterCallback FileWriter,
                             void *File, uint8_t *Buffer, uint32_t BufferSz) {
  BufferIO->File = File;
  BufferIO->FileWriter = FileWriter;
  BufferIO->BufferStart = Buffer;
  BufferIO->BufferSz = BufferSz;
  BufferIO->CurOffset = 0;
}

COMPILER_RT_VISIBILITY ProfBufferIO *
llvmCreateBufferIO(WriterCallback FileWriter, void *File, uint32_t BufferSz) {
  ProfBufferIO *BufferIO = (ProfBufferIO *)CallocHook(1, sizeof(ProfBufferIO));
  uint8_t *Buffer = (uint8_t *)CallocHook(1, BufferSz);
  if (!Buffer) {
    FreeHook(BufferIO);
    return 0;
  }
  llvmInitBufferIO(BufferIO, FileWriter, File, Buffer, BufferSz);
  return BufferIO;
}

COMPILER_RT_VISIBILITY void llvmDeleteBufferIO(ProfBufferIO *BufferIO) {
  FreeHook(BufferIO->BufferStart);
  FreeHook(BufferIO);
}

COMPILER_RT_VISIBILITY int
llvmBufferIOWrite(ProfBufferIO *BufferIO, const uint8_t *Data, uint32_t Size) {
  /* Buffer is not large enough, it is time to flush.  */
  if (Size + BufferIO->CurOffset > BufferIO->BufferSz) {
     if (llvmBufferIOFlush(BufferIO) != 0)
       return -1;
  }
  /* Special case, bypass the buffer completely. */
  ProfDataIOVec IO[] = {{Data, sizeof(uint8_t), Size}};
  if (Size > BufferIO->BufferSz) {
    if (BufferIO->FileWriter(IO, 1, &BufferIO->File))
      return -1;
  } else {
    /* Write the data to buffer */
    uint8_t *Buffer = BufferIO->BufferStart + BufferIO->CurOffset;
    llvmBufferWriter(IO, 1, (void **)&Buffer);
    BufferIO->CurOffset = Buffer - BufferIO->BufferStart;
  }
  return 0;
}

COMPILER_RT_VISIBILITY int llvmBufferIOFlush(ProfBufferIO *BufferIO) {
  if (BufferIO->CurOffset) {
    ProfDataIOVec IO[] = {
        {BufferIO->BufferStart, sizeof(uint8_t), BufferIO->CurOffset}};
    if (BufferIO->FileWriter(IO, 1, &BufferIO->File))
      return -1;
    BufferIO->CurOffset = 0;
  }
  return 0;
}

COMPILER_RT_VISIBILITY int llvmWriteProfData(WriterCallback Writer,
                                             void *WriterCtx,
                                             ValueProfData **ValueDataArray,
                                             const uint64_t ValueDataSize) {
  /* Match logic in __llvm_profile_write_buffer(). */
  const __llvm_profile_data *DataBegin = __llvm_profile_begin_data();
  const __llvm_profile_data *DataEnd = __llvm_profile_end_data();
  const uint64_t *CountersBegin = __llvm_profile_begin_counters();
  const uint64_t *CountersEnd = __llvm_profile_end_counters();
  const char *NamesBegin = __llvm_profile_begin_names();
  const char *NamesEnd = __llvm_profile_end_names();
  return llvmWriteProfDataImpl(Writer, WriterCtx, DataBegin, DataEnd,
                               CountersBegin, CountersEnd, ValueDataArray,
                               ValueDataSize, NamesBegin, NamesEnd);
}

#define VP_BUFFER_SIZE 8 * 1024
static int writeValueProfData(WriterCallback Writer, void *WriterCtx,
                              ValueProfData **ValueDataBegin,
                              uint64_t NumVData) {
  ProfBufferIO *BufferIO;
  uint32_t I = 0, BufferSz;

  if (!ValueDataBegin)
    return 0;

  BufferSz = VPBufferSize ? VPBufferSize : VP_BUFFER_SIZE;
  BufferIO = llvmCreateBufferIO(Writer, WriterCtx, BufferSz);

  for (I = 0; I < NumVData; I++) {
    ValueProfData *CurVData = ValueDataBegin[I];
    if (!CurVData)
      continue;
    if (llvmBufferIOWrite(BufferIO, (const uint8_t *)CurVData,
                          CurVData->TotalSize) != 0)
      return -1;
  }

  if (llvmBufferIOFlush(BufferIO) != 0)
    return -1;
  llvmDeleteBufferIO(BufferIO);

  return 0;
}

COMPILER_RT_VISIBILITY int llvmWriteProfDataImpl(
    WriterCallback Writer, void *WriterCtx,
    const __llvm_profile_data *DataBegin, const __llvm_profile_data *DataEnd,
    const uint64_t *CountersBegin, const uint64_t *CountersEnd,
    ValueProfData **ValueDataBegin, const uint64_t ValueDataSize,
    const char *NamesBegin, const char *NamesEnd) {

  /* Calculate size of sections. */
  const uint64_t DataSize = DataEnd - DataBegin;
  const uint64_t CountersSize = CountersEnd - CountersBegin;
  const uint64_t NamesSize = NamesEnd - NamesBegin;
  const uint64_t Padding = __llvm_profile_get_num_padding_bytes(NamesSize);

  /* Enough zeroes for padding. */
  const char Zeroes[sizeof(uint64_t)] = {0};

  /* Create the header. */
  __llvm_profile_header Header;

  if (!DataSize)
    return 0;

  /* Initialize header struture.  */
#define INSTR_PROF_RAW_HEADER(Type, Name, Init) Header.Name = Init;
#include "InstrProfData.inc"

  /* Write the data. */
  ProfDataIOVec IOVec[] = {{&Header, sizeof(__llvm_profile_header), 1},
                           {DataBegin, sizeof(__llvm_profile_data), DataSize},
                           {CountersBegin, sizeof(uint64_t), CountersSize},
                           {NamesBegin, sizeof(uint8_t), NamesSize},
                           {Zeroes, sizeof(uint8_t), Padding}};
  if (Writer(IOVec, sizeof(IOVec) / sizeof(*IOVec), &WriterCtx))
    return -1;

  return writeValueProfData(Writer, WriterCtx, ValueDataBegin, DataSize);
}
