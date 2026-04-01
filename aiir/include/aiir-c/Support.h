//===-- aiir-c/Support.h - Helpers for C API to Core AIIR ---------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the auxiliary data structures used in C APIs to core
// AIIR functionality.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_SUPPORT_H
#define AIIR_C_SUPPORT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

//===----------------------------------------------------------------------===//
// Visibility annotations.
// Use AIIR_CAPI_EXPORTED for exported functions.
//
// On Windows, if AIIR_CAPI_ENABLE_WINDOWS_DLL_DECLSPEC is defined, then
// __declspec(dllexport) and __declspec(dllimport) will be generated. This
// can only be enabled if actually building DLLs. It is generally, mutually
// exclusive with the use of other mechanisms for managing imports/exports
// (i.e. CMake's WINDOWS_EXPORT_ALL_SYMBOLS feature).
//===----------------------------------------------------------------------===//

#if (defined(_WIN32) || defined(__CYGWIN__)) &&                                \
    !defined(AIIR_CAPI_ENABLE_WINDOWS_DLL_DECLSPEC)
// Visibility annotations disabled.
#define AIIR_CAPI_EXPORTED
#elif defined(_WIN32) || defined(__CYGWIN__)
// Windows visibility declarations.
#if AIIR_CAPI_BUILDING_LIBRARY
#define AIIR_CAPI_EXPORTED __declspec(dllexport)
#else
#define AIIR_CAPI_EXPORTED __declspec(dllimport)
#endif
#else
// Non-windows: use visibility attributes.
#define AIIR_CAPI_EXPORTED __attribute__((visibility("default")))
#endif

#define AIIR_PYTHON_API_EXPORTED AIIR_CAPI_EXPORTED

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

/// Re-export llvm::ThreadPool so as to avoid including the LLVM C API directly.
DEFINE_C_API_STRUCT(AiirLlvmThreadPool, void);
/// Re-export llvm::raw_fd_ostream so as to avoid including the LLVM C API
/// directly.
DEFINE_C_API_STRUCT(AiirLlvmRawFdOStream, void);
DEFINE_C_API_STRUCT(AiirTypeID, const void);
DEFINE_C_API_STRUCT(AiirTypeIDAllocator, void);

#undef DEFINE_C_API_STRUCT

//===----------------------------------------------------------------------===//
// AiirStringRef.
//===----------------------------------------------------------------------===//

/// A pointer to a sized fragment of a string, not necessarily null-terminated.
/// Does not own the underlying string. This is equivalent to llvm::StringRef.

struct AiirStringRef {
  const char *data; ///< Pointer to the first symbol.
  size_t length;    ///< Length of the fragment.
};
typedef struct AiirStringRef AiirStringRef;

/// Constructs a string reference from the pointer and length. The pointer need
/// not reference to a null-terminated string.

inline static AiirStringRef aiirStringRefCreate(const char *str,
                                                size_t length) {
  AiirStringRef result;
  result.data = str;
  result.length = length;
  return result;
}

/// Constructs a string reference from a null-terminated C string. Prefer
/// aiirStringRefCreate if the length of the string is known.
AIIR_CAPI_EXPORTED AiirStringRef
aiirStringRefCreateFromCString(const char *str);

/// Returns true if two string references are equal, false otherwise.
AIIR_CAPI_EXPORTED bool aiirStringRefEqual(AiirStringRef string,
                                           AiirStringRef other);

/// A callback for returning string references.
///
/// This function is called back by the functions that need to return a
/// reference to the portion of the string with the following arguments:
///  - an AiirStringRef representing the current portion of the string
///  - a pointer to user data forwarded from the printing call.
typedef void (*AiirStringCallback)(AiirStringRef, void *);

//===----------------------------------------------------------------------===//
// AiirLogicalResult.
//===----------------------------------------------------------------------===//

/// A logical result value, essentially a boolean with named states. LLVM
/// convention for using boolean values to designate success or failure of an
/// operation is a moving target, so AIIR opted for an explicit class.
/// Instances of AiirLogicalResult must only be inspected using the associated
/// functions.
struct AiirLogicalResult {
  int8_t value;
};
typedef struct AiirLogicalResult AiirLogicalResult;

/// Checks if the given logical result represents a success.
inline static bool aiirLogicalResultIsSuccess(AiirLogicalResult res) {
  return res.value != 0;
}

/// Checks if the given logical result represents a failure.
inline static bool aiirLogicalResultIsFailure(AiirLogicalResult res) {
  return res.value == 0;
}

/// Creates a logical result representing a success.
inline static AiirLogicalResult aiirLogicalResultSuccess(void) {
  AiirLogicalResult res = {1};
  return res;
}

/// Creates a logical result representing a failure.
inline static AiirLogicalResult aiirLogicalResultFailure(void) {
  AiirLogicalResult res = {0};
  return res;
}

//===----------------------------------------------------------------------===//
// AiirLlvmThreadPool.
//===----------------------------------------------------------------------===//

/// Create an LLVM thread pool. This is reexported here to avoid directly
/// pulling in the LLVM headers directly.
AIIR_CAPI_EXPORTED AiirLlvmThreadPool aiirLlvmThreadPoolCreate(void);

/// Destroy an LLVM thread pool.
AIIR_CAPI_EXPORTED void aiirLlvmThreadPoolDestroy(AiirLlvmThreadPool pool);

/// Returns the maximum number of threads in the thread pool.
AIIR_CAPI_EXPORTED int
aiirLlvmThreadPoolGetMaxConcurrency(AiirLlvmThreadPool pool);

//===----------------------------------------------------------------------===//
// AiirLlvmRawFdOStream.
//===----------------------------------------------------------------------===//

/// Create a raw_fd_ostream for the given path. This wrapper is needed because
/// std::ostream does not provide the file sharing semantics required on
/// Windows.
/// - `path`: output file path.
/// - `binary`: controls text vs binary mode.
/// - `errorCallback`: called with an error message on failure (optional).
/// - `userData`: forwarded to `errorCallback` so it can copy the error message
///   into caller-owned storage (e.g., a `std::string`).
/// On failure, returns a null stream and invokes the optional error callback
/// with the error message.
AIIR_CAPI_EXPORTED AiirLlvmRawFdOStream
aiirLlvmRawFdOStreamCreate(const char *path, bool binary,
                           AiirStringCallback errorCallback, void *userData);

/// Write a string to a raw_fd_ostream created with aiirLlvmRawFdOStreamCreate.
AIIR_CAPI_EXPORTED void aiirLlvmRawFdOStreamWrite(AiirLlvmRawFdOStream stream,
                                                  AiirStringRef string);

/// Checks if a raw_fd_ostream is null.
AIIR_CAPI_EXPORTED bool aiirLlvmRawFdOStreamIsNull(AiirLlvmRawFdOStream stream);

/// Destroy a raw_fd_ostream created with aiirLlvmRawFdOStreamCreate.
AIIR_CAPI_EXPORTED void
aiirLlvmRawFdOStreamDestroy(AiirLlvmRawFdOStream stream);

//===----------------------------------------------------------------------===//
// TypeID API.
//===----------------------------------------------------------------------===//

/// `ptr` must be 8 byte aligned and unique to a type valid for the duration of
/// the returned type id's usage
AIIR_CAPI_EXPORTED AiirTypeID aiirTypeIDCreate(const void *ptr);

/// Checks whether a type id is null.
static inline bool aiirTypeIDIsNull(AiirTypeID typeID) { return !typeID.ptr; }

/// Checks if two type ids are equal.
AIIR_CAPI_EXPORTED bool aiirTypeIDEqual(AiirTypeID typeID1, AiirTypeID typeID2);

/// Returns the hash value of the type id.
AIIR_CAPI_EXPORTED size_t aiirTypeIDHashValue(AiirTypeID typeID);

//===----------------------------------------------------------------------===//
// TypeIDAllocator API.
//===----------------------------------------------------------------------===//

/// Creates a type id allocator for dynamic type id creation
AIIR_CAPI_EXPORTED AiirTypeIDAllocator aiirTypeIDAllocatorCreate(void);

/// Deallocates the allocator and all allocated type ids
AIIR_CAPI_EXPORTED void
aiirTypeIDAllocatorDestroy(AiirTypeIDAllocator allocator);

/// Allocates a type id that is valid for the lifetime of the allocator
AIIR_CAPI_EXPORTED AiirTypeID
aiirTypeIDAllocatorAllocateTypeID(AiirTypeIDAllocator allocator);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_SUPPORT_H
