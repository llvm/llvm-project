//===-- InstrumentationRuntime.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_INSTRUMENTATIONRUNTIME_H
#define LLDB_TARGET_INSTRUMENTATIONRUNTIME_H

#include <map>
#include <vector>

#include "lldb/Core/PluginInterface.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

typedef std::map<lldb::InstrumentationRuntimeType,
                 lldb::InstrumentationRuntimeSP>
    InstrumentationRuntimeCollection;

/// \class InstrumentationRuntime InstrumentationRuntime.h
/// "lldb/Target/InstrumentationRuntime.h"
///
/// InstrumentationRuntime plugins detect and interact with runtime
/// instrumentation libraries that are injected into the debugged process.
/// These libraries perform dynamic analysis such as detecting memory errors,
/// race conditions, or other runtime issues.
///
/// This plugin type represents runtime sanitizers and checkers, including:
/// - AddressSanitizer (ASan): Detects memory errors like buffer overflows,
///   use-after-free, and memory leaks
/// - ThreadSanitizer (TSan): Detects data races and threading issues
/// - UndefinedBehaviorSanitizer (UBSan): Detects undefined behavior like
///   integer overflow or null pointer dereference
/// - BoundsSafety: Detects out-of-bounds accesses
/// - MainThreadChecker: Detects violations of main thread requirements (macOS)
///
/// How LLDB Uses InstrumentationRuntime Plugins:
///
/// 1. Detection Phase (ModulesDidLoad):
///    When new dynamic libraries are loaded into the process, LLDB calls
///    InstrumentationRuntime::ModulesDidLoad() with the module list. This
///    static method:
///    - Queries the PluginManager for all registered InstrumentationRuntime
///      plugins via their create_callback functions
///    - Creates one instance of each plugin type for the process
///    - Each instance checks if its corresponding runtime library is loaded
///
/// 2. Runtime Library Identification:
///    Each plugin instance examines loaded modules to find its runtime library:
///    - GetPatternForRuntimeLibrary() returns a regex matching the library name
///      (e.g., "libclang_rt.asan.*" for ASan)
///    - CheckIfRuntimeIsValid() performs additional validation on candidate
///      modules (e.g., checking for required symbols)
///    - MatchAllModules() returns true if all modules should be checked, not
///      just those matching the regex pattern
///
/// 3. Activation Phase:
///    Once the runtime library is identified and validated:
///    - SetRuntimeModuleSP() caches the runtime module
///    - Activate() is called to set up interaction with the runtime:
///      * Typically sets breakpoints in the runtime library at reporting
///        functions (e.g., __asan_report_error)
///      * Stores the breakpoint ID in m_breakpoint_id
///      * Marks the runtime as active via SetActive(true)
///
/// 4. Runtime Event Handling:
///    When a breakpoint in the runtime library is hit:
///    - The runtime typically stores diagnostic information (stack traces,
///      error details) in memory structures
///    - GetBacktracesFromExtendedStopInfo() can extract this information and
///      create synthetic thread backtraces showing where the error occurred
///    - This allows LLDB to display rich diagnostics like "heap-use-after-free
///      on address 0x12345678, allocated by thread T0, freed by thread T1"
///
/// Important Implementation Considerations:
/// - Plugins are instantiated once per process, not per runtime library
/// - The IsActive() flag prevents duplicate activation
/// - Breakpoints set in Activate() should be stored in m_breakpoint_id for
///   later cleanup or management
/// - Runtime libraries may be loaded late (after process launch) or early
///   (before main), so ModulesDidLoad is called repeatedly as libraries load
/// - Multiple instrumentation runtimes can be active simultaneously (e.g.,
///   both ASan and UBSan)
/// - Implementations should be careful with symbols that may not be present
///   in all versions of the runtime library
/// - Extended stop info is passed as StructuredData and must be parsed
///   according to the runtime's specific format
class InstrumentationRuntime
    : public std::enable_shared_from_this<InstrumentationRuntime>,
      public PluginInterface {
  /// The instrumented process.
  lldb::ProcessWP m_process_wp;

  /// The module containing the instrumentation runtime.
  lldb::ModuleSP m_runtime_module;

  /// The breakpoint in the instrumentation runtime.
  lldb::user_id_t m_breakpoint_id;

  /// Indicates whether or not breakpoints have been registered in the
  /// instrumentation runtime.
  bool m_is_active;

protected:
  InstrumentationRuntime(const lldb::ProcessSP &process_sp)
      : m_breakpoint_id(0), m_is_active(false) {
    if (process_sp)
      m_process_wp = process_sp;
  }

  lldb::ProcessSP GetProcessSP() { return m_process_wp.lock(); }

  lldb::ModuleSP GetRuntimeModuleSP() { return m_runtime_module; }

  void SetRuntimeModuleSP(lldb::ModuleSP module_sp) {
    m_runtime_module = std::move(module_sp);
  }

  lldb::user_id_t GetBreakpointID() const { return m_breakpoint_id; }

  void SetBreakpointID(lldb::user_id_t ID) { m_breakpoint_id = ID; }

  void SetActive(bool IsActive) { m_is_active = IsActive; }

  /// Return a regular expression which can be used to identify a valid version
  /// of the runtime library.
  virtual const RegularExpression &GetPatternForRuntimeLibrary() = 0;

  /// Check whether \p module_sp corresponds to a valid runtime library.
  virtual bool CheckIfRuntimeIsValid(const lldb::ModuleSP module_sp) = 0;

  /// Register a breakpoint in the runtime library and perform any other
  /// necessary initialization. The runtime library
  /// is guaranteed to be loaded.
  virtual void Activate() = 0;

  /// \return true if `CheckIfRuntimeIsValid` should be called on all modules.
  /// In this case the return value of `GetPatternForRuntimeLibrary` will be
  /// ignored. Return false if `CheckIfRuntimeIsValid` should only be called
  /// for modules whose name matches `GetPatternForRuntimeLibrary`.
  ///
  virtual bool MatchAllModules() { return false; }

public:
  static void ModulesDidLoad(lldb_private::ModuleList &module_list,
                             Process *process,
                             InstrumentationRuntimeCollection &runtimes);

  /// Look for the instrumentation runtime in \p module_list. Register and
  /// activate the runtime if this hasn't already
  /// been done.
  void ModulesDidLoad(lldb_private::ModuleList &module_list);

  bool IsActive() const { return m_is_active; }

  virtual lldb::ThreadCollectionSP
  GetBacktracesFromExtendedStopInfo(StructuredData::ObjectSP info);
};

} // namespace lldb_private

#endif // LLDB_TARGET_INSTRUMENTATIONRUNTIME_H
