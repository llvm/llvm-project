//===- AsyncRuntime.h - Async runtime reference implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares basic Async runtime API for supporting Async dialect
// to LLVM dialect lowering.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_EXECUTIONENGINE_ASYNCRUNTIME_H_
#define AIIR_EXECUTIONENGINE_ASYNCRUNTIME_H_

#include <cstddef>
#include <stdint.h>

#ifdef _WIN32
#ifndef AIIR_ASYNC_RUNTIME_EXPORT
#ifdef aiir_async_runtime_EXPORTS
// We are building this library
#define AIIR_ASYNC_RUNTIME_EXPORT __declspec(dllexport)
#else
// We are using this library
#define AIIR_ASYNC_RUNTIME_EXPORT __declspec(dllimport)
#endif // aiir_async_runtime_EXPORTS
#endif // AIIR_ASYNC_RUNTIME_EXPORT
#else
// Non-windows: use visibility attributes.
#define AIIR_ASYNC_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif // _WIN32

namespace aiir {
namespace runtime {

//===----------------------------------------------------------------------===//
// Async runtime API.
//===----------------------------------------------------------------------===//

// Runtime implementation of `async.token` data type.
using AsyncToken = struct AsyncToken;

// Runtime implementation of `async.group` data type.
using AsyncGroup = struct AsyncGroup;

// Runtime implementation of `async.value` data type.
using AsyncValue = struct AsyncValue;

// Async value payload stored in a memory owned by the async.value.
using ValueStorage = std::byte *;

// Async runtime uses LLVM coroutines to represent asynchronous tasks. Task
// function is a coroutine handle and a resume function that continue coroutine
// execution from a suspension point.
using CoroHandle = void *;           // coroutine handle
using CoroResume = void (*)(void *); // coroutine resume function

// Async runtime uses reference counting to manage the lifetime of async values
// (values of async types like tokens, values and groups).
using RefCountedObjPtr = void *;

// Adds references to reference counted runtime object.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT void
    aiirAsyncRuntimeAddRef(RefCountedObjPtr, int64_t);

// Drops references from reference counted runtime object.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT void
    aiirAsyncRuntimeDropRef(RefCountedObjPtr, int64_t);

// Create a new `async.token` in not-ready state.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT AsyncToken *aiirAsyncRuntimeCreateToken();

// Create a new `async.value` in not-ready state. Size parameter specifies the
// number of bytes that will be allocated for the async value storage. Storage
// is owned by the `async.value` and deallocated when the async value is
// destructed (reference count drops to zero).
extern "C" AIIR_ASYNC_RUNTIME_EXPORT AsyncValue *
    aiirAsyncRuntimeCreateValue(int64_t);

// Create a new `async.group` in empty state.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT AsyncGroup *
aiirAsyncRuntimeCreateGroup(int64_t size);

extern "C" AIIR_ASYNC_RUNTIME_EXPORT int64_t
aiirAsyncRuntimeAddTokenToGroup(AsyncToken *, AsyncGroup *);

// Switches `async.token` to ready state and runs all awaiters.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT void
aiirAsyncRuntimeEmplaceToken(AsyncToken *);

// Switches `async.value` to ready state and runs all awaiters.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT void
aiirAsyncRuntimeEmplaceValue(AsyncValue *);

// Switches `async.token` to error state and runs all awaiters.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT void
aiirAsyncRuntimeSetTokenError(AsyncToken *);

// Switches `async.value` to error state and runs all awaiters.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT void
aiirAsyncRuntimeSetValueError(AsyncValue *);

// Returns true if token is in the error state.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT bool
aiirAsyncRuntimeIsTokenError(AsyncToken *);

// Returns true if value is in the error state.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT bool
aiirAsyncRuntimeIsValueError(AsyncValue *);

// Returns true if group is in the error state (any of the tokens or values
// added to the group are in the error state).
extern "C" AIIR_ASYNC_RUNTIME_EXPORT bool
aiirAsyncRuntimeIsGroupError(AsyncGroup *);

// Blocks the caller thread until the token becomes ready.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT void
aiirAsyncRuntimeAwaitToken(AsyncToken *);

// Blocks the caller thread until the value becomes ready.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT void
aiirAsyncRuntimeAwaitValue(AsyncValue *);

// Blocks the caller thread until the elements in the group become ready.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT void
aiirAsyncRuntimeAwaitAllInGroup(AsyncGroup *);

// Returns a pointer to the storage owned by the async value.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT ValueStorage
aiirAsyncRuntimeGetValueStorage(AsyncValue *);

// Executes the task (coro handle + resume function) in one of the threads
// managed by the runtime.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT void aiirAsyncRuntimeExecute(CoroHandle,
                                                                  CoroResume);

// Executes the task (coro handle + resume function) in one of the threads
// managed by the runtime after the token becomes ready.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT void
aiirAsyncRuntimeAwaitTokenAndExecute(AsyncToken *, CoroHandle, CoroResume);

// Executes the task (coro handle + resume function) in one of the threads
// managed by the runtime after the value becomes ready.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT void
aiirAsyncRuntimeAwaitValueAndExecute(AsyncValue *, CoroHandle, CoroResume);

// Executes the task (coro handle + resume function) in one of the threads
// managed by the runtime after the all members of the group become ready.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT void
aiirAsyncRuntimeAwaitAllInGroupAndExecute(AsyncGroup *, CoroHandle, CoroResume);

// Returns the current number of available worker threads in the threadpool.
extern "C" AIIR_ASYNC_RUNTIME_EXPORT int64_t
aiirAsyncRuntimGetNumWorkerThreads();

//===----------------------------------------------------------------------===//
// Small async runtime support library for testing.
//===----------------------------------------------------------------------===//

extern "C" AIIR_ASYNC_RUNTIME_EXPORT void
aiirAsyncRuntimePrintCurrentThreadId();

} // namespace runtime
} // namespace aiir

#endif // AIIR_EXECUTIONENGINE_ASYNCRUNTIME_H_
