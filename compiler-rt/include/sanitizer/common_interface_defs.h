//===-- sanitizer/common_interface_defs.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common part of the public sanitizer interface.
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_COMMON_INTERFACE_DEFS_H
#define SANITIZER_COMMON_INTERFACE_DEFS_H

#include <stddef.h>
#include <stdint.h>

// Windows allows a user to set their default calling convention, but we always
// use __cdecl
#ifdef _WIN32
#define SANITIZER_CDECL __cdecl
#else
#define SANITIZER_CDECL
#endif

#ifdef __cplusplus
extern "C" {
#endif
// Arguments for __sanitizer_sandbox_on_notify() below.
typedef struct {
  // Enable sandbox support in sanitizer coverage.
  int coverage_sandboxed;
  // File descriptor to write coverage data to. If -1 is passed, a file will
  // be pre-opened by __sanitizer_sandbox_on_notify(). This field has no
  // effect if coverage_sandboxed == 0.
  intptr_t coverage_fd;
  // If non-zero, split the coverage data into well-formed blocks. This is
  // useful when coverage_fd is a socket descriptor. Each block will contain
  // a header, allowing data from multiple processes to be sent over the same
  // socket.
  unsigned int coverage_max_block_size;
} __sanitizer_sandbox_arguments;

// Tell the tools to write their reports to "path.<pid>" instead of stderr.
void SANITIZER_CDECL __sanitizer_set_report_path(const char *path);
// Tell the tools to write their reports to the provided file descriptor
// (casted to void *).
void SANITIZER_CDECL __sanitizer_set_report_fd(void *fd);
// Get the current full report file path, if a path was specified by
// an earlier call to __sanitizer_set_report_path. Returns null otherwise.
const char *SANITIZER_CDECL __sanitizer_get_report_path();

// Notify the tools that the sandbox is going to be turned on. The reserved
// parameter will be used in the future to hold a structure with functions
// that the tools may call to bypass the sandbox.
void SANITIZER_CDECL
__sanitizer_sandbox_on_notify(__sanitizer_sandbox_arguments *args);

// This function is called by the tool when it has just finished reporting
// an error. 'error_summary' is a one-line string that summarizes
// the error message. This function can be overridden by the client.
void SANITIZER_CDECL
__sanitizer_report_error_summary(const char *error_summary);

// Some of the sanitizers (for example ASan/TSan) could miss bugs that happen
// in unaligned loads/stores. To find such bugs reliably, you need to replace
// plain unaligned loads/stores with these calls.

/// Loads a 16-bit unaligned value.
//
/// \param p Pointer to unaligned memory.
///
/// \returns Loaded value.
uint16_t SANITIZER_CDECL __sanitizer_unaligned_load16(const void *p);

/// Loads a 32-bit unaligned value.
///
/// \param p Pointer to unaligned memory.
///
/// \returns Loaded value.
uint32_t SANITIZER_CDECL __sanitizer_unaligned_load32(const void *p);

/// Loads a 64-bit unaligned value.
///
/// \param p Pointer to unaligned memory.
///
/// \returns Loaded value.
uint64_t SANITIZER_CDECL __sanitizer_unaligned_load64(const void *p);

/// Stores a 16-bit unaligned value.
///
/// \param p Pointer to unaligned memory.
/// \param x 16-bit value to store.
void SANITIZER_CDECL __sanitizer_unaligned_store16(void *p, uint16_t x);

/// Stores a 32-bit unaligned value.
///
/// \param p Pointer to unaligned memory.
/// \param x 32-bit value to store.
void SANITIZER_CDECL __sanitizer_unaligned_store32(void *p, uint32_t x);

/// Stores a 64-bit unaligned value.
///
/// \param p Pointer to unaligned memory.
/// \param x 64-bit value to store.
void SANITIZER_CDECL __sanitizer_unaligned_store64(void *p, uint64_t x);

// Returns 1 on the first call, then returns 0 thereafter.  Called by the tool
// to ensure only one report is printed when multiple errors occur
// simultaneously.
int SANITIZER_CDECL __sanitizer_acquire_crash_state();

// The container overflow function declarations used to be inline here, but in
// order to allow libcxx to directly use the header they are now in a separate
// file. This file is included here to avoid breaking anyone reliant on
// the definitions appearing in the current file.
#include "container_overflow_defs.h"

/// Prints the stack trace leading to this call (useful for calling from the
/// debugger).
void SANITIZER_CDECL __sanitizer_print_stack_trace(void);

// Symbolizes the supplied 'pc' using the format string 'fmt'.
// Outputs at most 'out_buf_size' bytes into 'out_buf'.
// If 'out_buf' is not empty then output is zero or more non empty C strings
// followed by single empty C string. Multiple strings can be returned if PC
// corresponds to inlined function. Inlined frames are printed in the order
// from "most-inlined" to the "least-inlined", so the last frame should be the
// not inlined function.
// Inlined frames can be removed with 'symbolize_inline_frames=0'.
// The format syntax is described in
// lib/sanitizer_common/sanitizer_stacktrace_printer.h.
void SANITIZER_CDECL __sanitizer_symbolize_pc(void *pc, const char *fmt,
                                              char *out_buf,
                                              size_t out_buf_size);
// Same as __sanitizer_symbolize_pc, but for data section (i.e. globals).
void SANITIZER_CDECL __sanitizer_symbolize_global(void *data_ptr,
                                                  const char *fmt,
                                                  char *out_buf,
                                                  size_t out_buf_size);
// Determine the return address.
#if !defined(_MSC_VER) || defined(__clang__)
#define __sanitizer_return_address()                                           \
  __builtin_extract_return_addr(__builtin_return_address(0))
#else
void *_ReturnAddress(void);
#pragma intrinsic(_ReturnAddress)
#define __sanitizer_return_address() _ReturnAddress()
#endif

/// Sets the callback to be called immediately before death on error.
///
/// Passing 0 will unset the callback.
///
/// \param callback User-provided callback.
void SANITIZER_CDECL __sanitizer_set_death_callback(void (*callback)(void));

// Interceptor hooks.
// Whenever a libc function interceptor is called, it checks if the
// corresponding weak hook is defined, and calls it if it is indeed defined.
// The primary use-case is data-flow-guided fuzzing, where the fuzzer needs
// to know what is being passed to libc functions (for example memcmp).
// FIXME: implement more hooks.

/// Interceptor hook for <c>memcmp()</c>.
///
/// \param called_pc PC (program counter) address of the original call.
/// \param s1 Pointer to block of memory.
/// \param s2 Pointer to block of memory.
/// \param n Number of bytes to compare.
/// \param result Value returned by the intercepted function.
void SANITIZER_CDECL __sanitizer_weak_hook_memcmp(void *called_pc,
                                                  const void *s1,
                                                  const void *s2, size_t n,
                                                  int result);

/// Interceptor hook for <c>strncmp()</c>.
///
/// \param called_pc PC (program counter) address of the original call.
/// \param s1 Pointer to block of memory.
/// \param s2 Pointer to block of memory.
/// \param n Number of bytes to compare.
/// \param result Value returned by the intercepted function.
void SANITIZER_CDECL __sanitizer_weak_hook_strncmp(void *called_pc,
                                                   const char *s1,
                                                   const char *s2, size_t n,
                                                   int result);

/// Interceptor hook for <c>strncasecmp()</c>.
///
/// \param called_pc PC (program counter) address of the original call.
/// \param s1 Pointer to block of memory.
/// \param s2 Pointer to block of memory.
/// \param n Number of bytes to compare.
/// \param result Value returned by the intercepted function.
void SANITIZER_CDECL __sanitizer_weak_hook_strncasecmp(void *called_pc,
                                                       const char *s1,
                                                       const char *s2, size_t n,
                                                       int result);

/// Interceptor hook for <c>strcmp()</c>.
///
/// \param called_pc PC (program counter) address of the original call.
/// \param s1 Pointer to block of memory.
/// \param s2 Pointer to block of memory.
/// \param result Value returned by the intercepted function.
void SANITIZER_CDECL __sanitizer_weak_hook_strcmp(void *called_pc,
                                                  const char *s1,
                                                  const char *s2, int result);

/// Interceptor hook for <c>strcasecmp()</c>.
///
/// \param called_pc PC (program counter) address of the original call.
/// \param s1 Pointer to block of memory.
/// \param s2 Pointer to block of memory.
/// \param result Value returned by the intercepted function.
void SANITIZER_CDECL __sanitizer_weak_hook_strcasecmp(void *called_pc,
                                                      const char *s1,
                                                      const char *s2,
                                                      int result);

/// Interceptor hook for <c>strstr()</c>.
///
/// \param called_pc PC (program counter) address of the original call.
/// \param s1 Pointer to block of memory.
/// \param s2 Pointer to block of memory.
/// \param result Value returned by the intercepted function.
void SANITIZER_CDECL __sanitizer_weak_hook_strstr(void *called_pc,
                                                  const char *s1,
                                                  const char *s2, char *result);

void SANITIZER_CDECL __sanitizer_weak_hook_strcasestr(void *called_pc,
                                                      const char *s1,
                                                      const char *s2,
                                                      char *result);

void SANITIZER_CDECL __sanitizer_weak_hook_memmem(void *called_pc,
                                                  const void *s1, size_t len1,
                                                  const void *s2, size_t len2,
                                                  void *result);

// Prints stack traces for all live heap allocations ordered by total
// allocation size until top_percent of total live heap is shown. top_percent
// should be between 1 and 100. At most max_number_of_contexts contexts
// (stack traces) are printed.
// Experimental feature currently available only with ASan on Linux/x86_64.
void SANITIZER_CDECL __sanitizer_print_memory_profile(
    size_t top_percent, size_t max_number_of_contexts);

/// Notify ASan that a fiber switch has started (required only if implementing
/// your own fiber library).
///
/// Before switching to a different stack, you must call
/// <c>__sanitizer_start_switch_fiber()</c> with a pointer to the bottom of the
/// destination stack and with its size. When code starts running on the new
/// stack, it must call <c>__sanitizer_finish_switch_fiber()</c> to finalize
/// the switch. The <c>__sanitizer_start_switch_fiber()</c> function takes a
/// <c>void**</c> pointer argument to store the current fake stack if there is
/// one (it is necessary when the runtime option
/// <c>detect_stack_use_after_return</c> is enabled).
///
/// When restoring a stack, this <c>void**</c> pointer must be given to the
/// <c>__sanitizer_finish_switch_fiber()</c> function. In most cases, this
/// pointer can be stored on the stack immediately before switching. When
/// leaving a fiber definitely, NULL must be passed as the first argument to
/// the <c>__sanitizer_start_switch_fiber()</c> function so that the fake stack
/// is destroyed. If your program does not need stack use-after-return
/// detection, you can always pass NULL to these two functions.
///
/// \note The fake stack mechanism is disabled during fiber switch, so if a
/// signal callback runs during the switch, it will not benefit from stack
/// use-after-return detection.
///
/// \param[out] fake_stack_save Fake stack save location.
/// \param bottom Bottom address of stack.
/// \param size Size of stack in bytes.
void SANITIZER_CDECL __sanitizer_start_switch_fiber(void **fake_stack_save,
                                                    const void *bottom,
                                                    size_t size);

/// Notify ASan that a fiber switch has completed (required only if
/// implementing your own fiber library).
///
/// When code starts running on the new stack, it must call
/// <c>__sanitizer_finish_switch_fiber()</c> to finalize
/// the switch. For usage details, see the description of
/// <c>__sanitizer_start_switch_fiber()</c>.
///
/// \param fake_stack_save Fake stack save location.
/// \param[out] bottom_old Bottom address of old stack.
/// \param[out] size_old Size of old stack in bytes.
void SANITIZER_CDECL __sanitizer_finish_switch_fiber(void *fake_stack_save,
                                                     const void **bottom_old,
                                                     size_t *size_old);

// Get full module name and calculate pc offset within it.
// Returns 1 if pc belongs to some module, 0 if module was not found.
int SANITIZER_CDECL __sanitizer_get_module_and_offset_for_pc(
    void *pc, char *module_path, size_t module_path_len, void **pc_offset);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SANITIZER_COMMON_INTERFACE_DEFS_H
