/*===- InstrProfiling.h- Support library for PGO instrumentation ----------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#ifndef PROFILE_INSTRPROFILING_INTERNALH_
#define PROFILE_INSTRPROFILING_INTERNALH_

#include "InstrProfiling.h"
#include "stddef.h"

/*!
 * \brief Write instrumentation data to the given buffer, given explicit
 * pointers to the live data in memory.  This function is probably not what you
 * want.  Use __llvm_profile_get_size_for_buffer instead.  Use this function if
 * your program has a custom memory layout.
 */
uint64_t __llvm_profile_get_size_for_buffer_internal(
    const __llvm_profile_data *DataBegin, const __llvm_profile_data *DataEnd,
    const uint64_t *CountersBegin, const uint64_t *CountersEnd,
    const char *NamesBegin, const char *NamesEnd);

/*!
 * \brief Write instrumentation data to the given buffer, given explicit
 * pointers to the live data in memory.  This function is probably not what you
 * want.  Use __llvm_profile_write_buffer instead.  Use this function if your
 * program has a custom memory layout.
 *
 * \pre \c Buffer is the start of a buffer at least as big as \a
 * __llvm_profile_get_size_for_buffer_internal().
 */
int __llvm_profile_write_buffer_internal(
    char *Buffer, const __llvm_profile_data *DataBegin,
    const __llvm_profile_data *DataEnd, const uint64_t *CountersBegin,
    const uint64_t *CountersEnd, const char *NamesBegin, const char *NamesEnd);

/*!
 * The data structure describing the data to be written by the
 * low level writer callback function.
 */
typedef struct ProfDataIOVec {
  const void *Data;
  size_t ElmSize;
  size_t NumElm;
} ProfDataIOVec;

typedef uint32_t (*WriterCallback)(ProfDataIOVec *, uint32_t NumIOVecs,
                                   void **WriterCtx);

/*!
 * The data structure for buffered IO of profile data.
 */
typedef struct ProfBufferIO {
  /* File handle.  */
  void *File;
  /* Low level IO callback. */
  WriterCallback FileWriter;
  /* The start of the buffer. */
  uint8_t *BufferStart;
  /* Total size of the buffer. */
  uint32_t BufferSz;
  /* Current byte offset from the start of the buffer. */
  uint32_t CurOffset;
} ProfBufferIO;

/* The creator interface used by testing.  */
ProfBufferIO *lprofCreateBufferIOInternal(void *File, uint32_t DefaultBufferSz);
/*!
 * This is the interface to create a handle for buffered IO.
 */
ProfBufferIO *lprofCreateBufferIO(WriterCallback FileWriter, void *File,
                                  uint32_t DefaultBufferSz);
/*!
 * The interface to destroy the bufferIO handle and reclaim
 * the memory.
 */
void lprofDeleteBufferIO(ProfBufferIO *BufferIO);

/*!
 * This is the interface to write \c Data of \c Size bytes through
 * \c BufferIO. Returns 0 if successful, otherwise return -1.
 */
int lprofBufferIOWrite(ProfBufferIO *BufferIO, const uint8_t *Data,
                       uint32_t Size);
/*!
 * The interface to flush the remaining data in the buffer.
 * through the low level writer callback.
 */
int lprofBufferIOFlush(ProfBufferIO *BufferIO);

/* The low level interface to write data into a buffer. It is used as the
 * callback by other high level writer methods such as buffered IO writer
 * and profile data writer.  */
uint32_t lprofBufferWriter(ProfDataIOVec *IOVecs, uint32_t NumIOVecs,
                           void **WriterCtx);

int lprofWriteData(WriterCallback Writer, void *WriterCtx,
                   struct ValueProfData **ValueDataArray,
                   const uint64_t ValueDataSize);
int lprofWriteDataImpl(WriterCallback Writer, void *WriterCtx,
                       const __llvm_profile_data *DataBegin,
                       const __llvm_profile_data *DataEnd,
                       const uint64_t *CountersBegin,
                       const uint64_t *CountersEnd,
                       struct ValueProfData **ValueDataBeginArray,
                       const uint64_t ValueDataSize, const char *NamesBegin,
                       const char *NamesEnd);
/* Merge value profile data pointed to by SrcValueProfData into
 * in-memory profile counters pointed by to DstData.  */
void lprofMergeValueProfData(struct ValueProfData *SrcValueProfData,
                             __llvm_profile_data *DstData);

extern char *(*GetEnvHook)(const char *);
extern void (*FreeHook)(void *);
extern void *(*CallocHook)(size_t, size_t);
extern void (*VPMergeHook)(struct ValueProfData *, __llvm_profile_data *);
extern uint32_t VPBufferSize;

#endif
