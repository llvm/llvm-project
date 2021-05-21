//===-- dfsan_interface.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DataFlowSanitizer.
//
// Public interface header.
//===----------------------------------------------------------------------===//
#ifndef DFSAN_INTERFACE_H
#define DFSAN_INTERFACE_H

#include <stddef.h>
#include <stdint.h>
#include <sanitizer/common_interface_defs.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint16_t dfsan_label;
typedef uint32_t dfsan_origin;

/// Stores information associated with a specific label identifier.  A label
/// may be a base label created using dfsan_create_label, with associated
/// text description and user data, or an automatically created union label,
/// which represents the union of two label identifiers (which may themselves
/// be base or union labels).
struct dfsan_label_info {
  // Fields for union labels, set to 0 for base labels.
  dfsan_label l1;
  dfsan_label l2;

  // Fields for base labels.
  const char *desc;
  void *userdata;
};

/// Signature of the callback argument to dfsan_set_write_callback().
typedef void (*dfsan_write_callback_t)(int fd, const void *buf, size_t count);

/// Computes the union of \c l1 and \c l2, possibly creating a union label in
/// the process.
dfsan_label dfsan_union(dfsan_label l1, dfsan_label l2);

/// Creates and returns a base label with the given description and user data.
dfsan_label dfsan_create_label(const char *desc, void *userdata);

/// Sets the label for each address in [addr,addr+size) to \c label.
void dfsan_set_label(dfsan_label label, void *addr, size_t size);

/// Sets the label for each address in [addr,addr+size) to the union of the
/// current label for that address and \c label.
void dfsan_add_label(dfsan_label label, void *addr, size_t size);

/// Retrieves the label associated with the given data.
///
/// The type of 'data' is arbitrary.  The function accepts a value of any type,
/// which can be truncated or extended (implicitly or explicitly) as necessary.
/// The truncation/extension operations will preserve the label of the original
/// value.
dfsan_label dfsan_get_label(long data);

/// Retrieves the immediate origin associated with the given data. The returned
/// origin may point to another origin.
///
/// The type of 'data' is arbitrary.
dfsan_origin dfsan_get_origin(long data);

/// Retrieves the label associated with the data at the given address.
dfsan_label dfsan_read_label(const void *addr, size_t size);

/// Retrieves a pointer to the dfsan_label_info struct for the given label.
const struct dfsan_label_info *dfsan_get_label_info(dfsan_label label);

/// Returns whether the given label label contains the label elem.
int dfsan_has_label(dfsan_label label, dfsan_label elem);

/// If the given label label contains a label with the description desc, returns
/// that label, else returns 0.
dfsan_label dfsan_has_label_with_desc(dfsan_label label, const char *desc);

/// Returns the number of labels allocated.
size_t dfsan_get_label_count(void);

/// Flushes the DFSan shadow, i.e. forgets about all labels currently associated
/// with the application memory.  Use this call to start over the taint tracking
/// within the same process.
///
/// Note: If another thread is working with tainted data during the flush, that
/// taint could still be written to shadow after the flush.
void dfsan_flush(void);

/// Sets a callback to be invoked on calls to write().  The callback is invoked
/// before the write is done.  The write is not guaranteed to succeed when the
/// callback executes.  Pass in NULL to remove any callback.
void dfsan_set_write_callback(dfsan_write_callback_t labeled_write_callback);

/// Writes the labels currently used by the program to the given file
/// descriptor. The lines of the output have the following format:
///
/// <label> <parent label 1> <parent label 2> <label description if any>
void dfsan_dump_labels(int fd);

/// Interceptor hooks.
/// Whenever a dfsan's custom function is called the corresponding
/// hook is called it non-zero. The hooks should be defined by the user.
/// The primary use case is taint-guided fuzzing, where the fuzzer
/// needs to see the parameters of the function and the labels.
/// FIXME: implement more hooks.
void dfsan_weak_hook_memcmp(void *caller_pc, const void *s1, const void *s2,
                            size_t n, dfsan_label s1_label,
                            dfsan_label s2_label, dfsan_label n_label);
void dfsan_weak_hook_strncmp(void *caller_pc, const char *s1, const char *s2,
                             size_t n, dfsan_label s1_label,
                             dfsan_label s2_label, dfsan_label n_label);

/// Prints the origin trace of the label at the address addr to stderr. It also
/// prints description at the beginning of the trace. If origin tracking is not
/// on, or the address is not labeled, it prints nothing.
void dfsan_print_origin_trace(const void *addr, const char *description);

/// Prints the origin trace of the label at the address \p addr to a
/// pre-allocated output buffer. If origin tracking is not on, or the address is
/// not labeled, it prints nothing.
///
/// Typical usage:
/// \code
///   char kDescription[] = "...";
///   char buf[1024];
///   dfsan_sprint_origin_trace(&tainted_var, kDescription, buf, sizeof(buf));
/// \endcode
///
/// Typical usage that handles truncation:
/// \code
///   char buf[1024];
///   int len = dfsan_sprint_origin_trace(&var, nullptr, buf, sizeof(buf));
///
///   if (len < sizeof(buf)) {
///     ProcessOriginTrace(tmpbuf);
///   else {
///     char *tmpbuf = new char[len + 1];
///     dfsan_sprint_origin_trace(&var, nullptr, tmpbuf, len + 1);
///     ProcessOriginTrace(tmpbuf);
///     delete[] tmpbuf;
///   }
/// \endcode
///
/// \param addr The tainted memory address whose origin we are printing.
/// \param description A description printed at the beginning of the trace.
/// \param [out] out_buf The output buffer to write the results to.
/// \param out_buf_size The size of \p out_buf.
///
/// \returns The number of symbols that should have been written to \p out_buf
/// (not including trailing null byte '\0'). Thus, the string is truncated iff
/// return value is not less than \p out_buf_size.
size_t dfsan_sprint_origin_trace(const void *addr, const char *description,
                                 char *out_buf, size_t out_buf_size);

/// Retrieves the very first origin associated with the data at the given
/// address.
dfsan_origin dfsan_get_init_origin(const void *addr);
#ifdef __cplusplus
}  // extern "C"

template <typename T>
void dfsan_set_label(dfsan_label label, T &data) { // NOLINT
  dfsan_set_label(label, (void *)&data, sizeof(T));
}

#endif

#endif  // DFSAN_INTERFACE_H
