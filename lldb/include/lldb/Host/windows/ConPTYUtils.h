//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_WINDOWS_CONPTYUTILS_H
#define LLDB_HOST_WINDOWS_CONPTYUTILS_H

#include <cstddef>

namespace lldb_private {

/// Remove ConPTY management sequences from a buffer in-place.
///
/// ConPTY injects several VT sequences into its output pipe that are not part
/// of the inferior's output: a cursor-position query (\x1b[6n), Win32 Input
/// Mode toggles (\x1b[?9001h/l), focus-event toggles (\x1b[?1004h/l), and a
/// window-title OSC sequence (\x1b]0;...\x07).
///
/// \param[in,out] data  Buffer containing raw ConPTY output.
/// \param[in,out] len   On entry, the number of valid bytes in \p data.
///                      Updated to the number of bytes after stripping.
/// \param[in] strip_init  If true, also strip init-only sequences (\x1b[m,
///                        \x1b[?25h) that ConPTY emits at startup.
void StripConPTYSequences(void *data, size_t &len, bool strip_init);

} // namespace lldb_private

#endif // LLDB_HOST_WINDOWS_CONPTYUTILS_H
