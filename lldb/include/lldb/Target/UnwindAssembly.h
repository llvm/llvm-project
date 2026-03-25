//===-- UnwindAssembly.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_UNWINDASSEMBLY_H
#define LLDB_TARGET_UNWINDASSEMBLY_H

#include "lldb/Core/PluginInterface.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

/// A plug-in interface definition class for assembly-based stack unwinding.
///
/// UnwindAssembly plugins provide architecture-specific instruction analysis
/// to generate stack unwinding information by examining function assembly code.
/// This is essential when debug information (like DWARF .eh_frame or .debug_frame)
/// is missing, incomplete, or doesn't cover all code paths.
///
/// These plugins perform several critical functions:
/// - Analyzing function prologues/epilogues to determine how the stack frame
///   is set up and torn down
/// - Creating UnwindPlans that describe how to restore registers at each
///   instruction in a function
/// - Identifying the first non-prologue instruction for breakpoint placement
/// - Augmenting existing unwind plans with additional information from assembly
///
/// LLDB instantiates UnwindAssembly plugins via UnwindAssembly::FindPlugin()
/// when unwinding is needed. The selection is based on the target architecture
/// (ArchSpec). Each architecture typically has one UnwindAssembly implementation
/// that understands its specific calling conventions and instruction encodings.
///
/// Key methods to implement:
/// - GetNonCallSiteUnwindPlanFromAssembly(): Create a complete unwind plan by
///   analyzing function assembly (used when no debug info is available)
/// - AugmentUnwindPlanFromCallSite(): Enhance an existing unwind plan with
///   information gleaned from assembly analysis
/// - GetFastUnwindPlan(): Create a simplified unwind plan for performance-critical
///   unwinding (e.g., during sampling)
/// - FirstNonPrologueInsn(): Find where the function prologue ends
///
/// Implementation notes:
/// - Plugins must understand architecture-specific instruction encodings
/// - Must handle various calling conventions (e.g., ARM's multiple frame pointer modes)
/// - Should be conservative when uncertain (invalid unwind is worse than no unwind)
/// - Performance matters as this code runs during every stack walk
/// - Used heavily for release builds without debug info and for system libraries
class UnwindAssembly : public std::enable_shared_from_this<UnwindAssembly>,
                       public PluginInterface {
public:
  static lldb::UnwindAssemblySP FindPlugin(const ArchSpec &arch);

  virtual bool
  GetNonCallSiteUnwindPlanFromAssembly(AddressRange &func, Thread &thread,
                                       UnwindPlan &unwind_plan) = 0;

  virtual bool AugmentUnwindPlanFromCallSite(AddressRange &func, Thread &thread,
                                             UnwindPlan &unwind_plan) = 0;

  virtual bool GetFastUnwindPlan(AddressRange &func, Thread &thread,
                                 UnwindPlan &unwind_plan) = 0;

  // thread may be NULL in which case we only use the Target (e.g. if this is
  // called pre-process-launch).
  virtual bool
  FirstNonPrologueInsn(AddressRange &func,
                       const lldb_private::ExecutionContext &exe_ctx,
                       Address &first_non_prologue_insn) = 0;

protected:
  UnwindAssembly(const ArchSpec &arch);
  ArchSpec m_arch;
};

} // namespace lldb_private

#endif // LLDB_TARGET_UNWINDASSEMBLY_H
