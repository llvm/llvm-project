/*===- c_api.h - C API for the ORC runtime ------------------------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file defines the C API for the ORC runtime                            *|
|*                                                                            *|
|* NOTE: The OrtRTWrapperFunctionResult type must be kept in sync with the    *|
|* definition in llvm/include/llvm-c/OrcShared.h.                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_API_H
#define ORC_RT_C_API_H

#include <assert.h>
#include <stdio.h>
#include <string.h>

/* Helper to suppress strict prototype warnings. */
#ifdef __clang__
#define ORC_RT_C_STRICT_PROTOTYPES_BEGIN                                       \
  _Pragma("clang diagnostic push")                                             \
      _Pragma("clang diagnostic error \"-Wstrict-prototypes\"")
#define ORC_RT_C_STRICT_PROTOTYPES_END _Pragma("clang diagnostic pop")
#else
#define ORC_RT_C_STRICT_PROTOTYPES_BEGIN
#define ORC_RT_C_STRICT_PROTOTYPES_END
#endif

/* Helper to wrap C code for C++ */
#ifdef __cplusplus
#define ORC_RT_C_EXTERN_C_BEGIN                                                \
  extern "C" {                                                                 \
  ORC_RT_C_STRICT_PROTOTYPES_BEGIN
#define ORC_RT_C_EXTERN_C_END                                                  \
  ORC_RT_C_STRICT_PROTOTYPES_END                                               \
  }
#else
#define ORC_RT_C_EXTERN_C_BEGIN ORC_RT_C_STRICT_PROTOTYPES_BEGIN
#define ORC_RT_C_EXTERN_C_END ORC_RT_C_STRICT_PROTOTYPES_END
#endif

ORC_RT_C_EXTERN_C_BEGIN

typedef union {
  const char *ValuePtr;
  char Value[sizeof(ValuePtr)];
} OrcRTCWrapperFunctionResultDataUnion;

/**
 * OrcRTCWrapperFunctionResult is a kind of C-SmallVector with an out-of-band
 * error state.
 *
 * If Size == 0 and Data.ValuePtr is non-zero then the value is in the
 * 'out-of-band error' state, and Data.ValuePtr points at a malloc-allocated,
 * null-terminated string error message.
 *
 * If Size <= sizeof(OrcRTCWrapperFunctionResultData) then the value is in the
 * 'small' state and the content is held in the first Size bytes of Data.Value.
 *
 * If Size > sizeof(OrtRTCWrapperFunctionResultData) then the value is in the
 * 'large' state and the content is held in the first Size bytes of the
 * memory pointed to by Data.ValuePtr. This memory must have been allocated by
 * malloc, and will be freed with free when this value is destroyed.
 */
typedef struct {
  OrcRTCWrapperFunctionResultDataUnion Data;
  size_t Size;
} OrcRTCWrapperFunctionResult;

typedef struct OrcRTCSharedOpaqueJITProcessControl
    *OrcRTSharedJITProcessControlRef;

/**
 * Zero-initialize an OrcRTCWrapperFunctionResult.
 */
static inline void
OrcRTCWrapperFunctionResultInit(OrcRTCWrapperFunctionResult *R) {
  R->Size = 0;
  R->Data.ValuePtr = 0;
}

/**
 * Create an OrcRTCWrapperFunctionResult with an uninitialized buffer of size
 * Size. The buffer is returned via the DataPtr argument.
 */
static inline char *
OrcRTCWrapperFunctionResultAllocate(OrcRTCWrapperFunctionResult *R,
                                    size_t Size) {
  char *DataPtr;
  R->Size = Size;
  if (Size > sizeof(R->Data.Value)) {
    DataPtr = (char *)malloc(Size);
    R->Data.ValuePtr = DataPtr;
  } else
    DataPtr = R->Data.Value;
  return DataPtr;
}

/**
 * Create an OrcRTWrapperFunctionResult from the given data range.
 */
static inline OrcRTCWrapperFunctionResult
OrcRTCreateCWrapperFunctionResultFromRange(const char *Data, size_t Size) {
  OrcRTCWrapperFunctionResult R;
  R.Size = Size;
  if (R.Size > sizeof(R.Data.Value)) {
    char *Tmp = (char *)malloc(Size);
    memcpy(Tmp, Data, Size);
    R.Data.ValuePtr = Tmp;
  } else
    memcpy(R.Data.Value, Data, Size);
  return R;
}

/**
 * Create an OrcRTCWrapperFunctionResult by copying the given string, including
 * the null-terminator.
 *
 * This function copies the input string. The client is responsible for freeing
 * the ErrMsg arg.
 */
static inline OrcRTCWrapperFunctionResult
OrcRTCreateCWrapperFunctionResultFromString(const char *Source) {
  return OrcRTCreateCWrapperFunctionResultFromRange(Source, strlen(Source) + 1);
}

/**
 * Create an OrcRTCWrapperFunctionResult representing an out-of-band
 * error.
 *
 * This function takes ownership of the string argument which must have been
 * allocated with malloc.
 */
static inline OrcRTCWrapperFunctionResult
OrcRTCreateCWrapperFunctionResultFromOutOfBandError(const char *ErrMsg) {
  OrcRTCWrapperFunctionResult R;
  R.Size = 0;
  char *Tmp = (char *)malloc(strlen(ErrMsg) + 1);
  strcpy(Tmp, ErrMsg);
  R.Data.ValuePtr = Tmp;
  return R;
}

/**
 * This should be called to destroy OrcRTCWrapperFunctionResult values
 * regardless of their state.
 */
static inline void
OrcRTDisposeCWrapperFunctionResult(OrcRTCWrapperFunctionResult *R) {
  if (R->Size > sizeof(R->Data.Value) ||
      (R->Size == 0 && R->Data.ValuePtr))
    free((void *)R->Data.ValuePtr);
}

/**
 * Get a pointer to the data contained in the given
 * OrcRTCWrapperFunctionResult.
 */
static inline const char *
OrcRTCWrapperFunctionResultData(const OrcRTCWrapperFunctionResult *R) {
  assert((R->Size != 0 || R->Data.ValuePtr == nullptr) &&
         "Cannot get data for out-of-band error value");
  return R->Size > sizeof(R->Data.Value) ? R->Data.ValuePtr : R->Data.Value;
}

/**
 * Safely get the size of the given OrcRTCWrapperFunctionResult.
 *
 * Asserts that we're not trying to access the size of an error value.
 */
static inline size_t
OrcRTCWrapperFunctionResultSize(const OrcRTCWrapperFunctionResult *R) {
  assert((R->Size != 0 || R->Data.ValuePtr == nullptr) &&
         "Cannot get size for out-of-band error value");
  return R->Size;
}

/**
 * Returns 1 if this value is equivalent to a value just initialized by
 * OrcRTCWrapperFunctionResultInit, 0 otherwise.
 */
static inline size_t
OrcRTCWrapperFunctionResultEmpty(const OrcRTCWrapperFunctionResult *R) {
  return R->Size == 0 && R->Data.ValuePtr == 0;
}

/**
 * Returns a pointer to the out-of-band error string for this
 * OrcRTCWrapperFunctionResult, or null if there is no error.
 *
 * The OrcRTCWrapperFunctionResult retains ownership of the error
 * string, so it should be copied if the caller wishes to preserve it.
 */
static inline const char *OrcRTCWrapperFunctionResultGetOutOfBandError(
    const OrcRTCWrapperFunctionResult *R) {
  return R->Size == 0 ? R->Data.ValuePtr : 0;
}

ORC_RT_C_EXTERN_C_END

#endif /* ORC_RT_C_API_H */
