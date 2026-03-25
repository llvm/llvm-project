//===-- MemoryHistory.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_MEMORYHISTORY_H
#define LLDB_TARGET_MEMORYHISTORY_H

#include <vector>

#include "lldb/Core/PluginInterface.h"
#include "lldb/lldb-private.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

typedef std::vector<lldb::ThreadSP> HistoryThreads;

/// \class MemoryHistory MemoryHistory.h "lldb/Target/MemoryHistory.h"
/// A plug-in interface definition class for memory history providers.
///
/// Memory history plugins track the allocation and access history of memory
/// addresses, which is crucial for debugging memory-related issues like
/// use-after-free, double-free, and buffer overflows. These plugins typically
/// integrate with runtime instrumentation tools (e.g., AddressSanitizer,
/// Malloc Stack Logging on Darwin) to provide historical stack traces showing
/// where memory was allocated, freed, or accessed.
///
/// When debugging memory corruption issues, LLDB can query memory history
/// plugins to get "history threads" - synthetic threads representing the
/// stack traces from previous operations on a memory address. This helps
/// developers understand what happened to memory before the current error.
///
/// Plugin Selection:
/// Memory history plugins are process-specific and created on-demand via
/// FindPlugin(). LLDB iterates through all registered memory history plugin
/// create callbacks (from PluginManager::GetMemoryHistoryCreateCallbacks())
/// until one returns a valid instance. Only one memory history plugin is
/// typically active per process, and it's created when first needed.
///
/// Key Responsibilities:
/// - GetHistoryThreads(): Return synthetic threads showing
/// allocation/deallocation
///   history for a given memory address
///
/// Important Notes:
/// - Plugins depend on runtime instrumentation being enabled in the target
/// process
/// - Must handle cases where no history is available for an address gracefully
/// - History threads are synthetic and don't represent actual execution threads
/// - Performance considerations: history lookup may involve reading significant
///   amounts of data from the target process
/// - The plugin should validate that the necessary runtime support is present
///   before attempting to retrieve history
class MemoryHistory : public std::enable_shared_from_this<MemoryHistory>,
                      public PluginInterface {
public:
  static lldb::MemoryHistorySP FindPlugin(const lldb::ProcessSP process);

  virtual HistoryThreads GetHistoryThreads(lldb::addr_t address) = 0;
};

} // namespace lldb_private

#endif // LLDB_TARGET_MEMORYHISTORY_H
