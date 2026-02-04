#ifndef LLDB_SYMBOL_FUNCUNWINDERS_H
#define LLDB_SYMBOL_FUNCUNWINDERS_H

#include "lldb/Core/AddressRange.h"
#include "lldb/lldb-private-enumerations.h"
#include <mutex>
#include <vector>

namespace lldb_private {

class UnwindTable;

class FuncUnwinders {
public:
  // FuncUnwinders objects are used to track UnwindPlans for a function (named
  // or not - really just a set of address ranges)

  // We'll record four different UnwindPlans for each function:
  //
  //   1. Unwinding from a call site (a valid exception throw location)
  //      This is often sourced from the eh_frame exception handling info
  //   2. Unwinding from a non-call site (any location in the function)
  //      This is often done by analyzing the function prologue assembly
  //      language instructions
  //   3. A fast unwind method for this function which only retrieves a
  //      limited set of registers necessary to walk the stack
  //   4. An architectural default unwind plan when none of the above are
  //      available for some reason.

  // Additionally, FuncUnwinds object can be asked where the prologue
  // instructions are finished for migrating breakpoints past the stack frame
  // setup instructions when we don't have line table information.

  FuncUnwinders(lldb_private::UnwindTable &unwind_table, Address addr,
                AddressRanges ranges);

  ~FuncUnwinders();

  std::shared_ptr<const UnwindPlan> GetUnwindPlanAtCallSite(Target &target,
                                                            Thread &thread);

  std::shared_ptr<const UnwindPlan>
  GetUnwindPlanAtNonCallSite(Target &target, lldb_private::Thread &thread);

  std::shared_ptr<const UnwindPlan>
  GetUnwindPlanFastUnwind(Target &target, lldb_private::Thread &thread);

  std::shared_ptr<const UnwindPlan>
  GetUnwindPlanArchitectureDefault(lldb_private::Thread &thread);

  std::shared_ptr<const UnwindPlan>
  GetUnwindPlanArchitectureDefaultAtFunctionEntry(lldb_private::Thread &thread);

  const Address &GetFunctionStartAddress() const;

  bool ContainsAddress(const Address &addr) const {
    return llvm::any_of(m_ranges, [&](const AddressRange range) {
      return range.ContainsFileAddress(addr);
    });
  }

  // The following methods to retrieve specific unwind plans should rarely be
  // used. Instead, clients should ask for the *behavior* they are looking for,
  // using one of the above UnwindPlan retrieval methods.

  std::shared_ptr<const UnwindPlan> GetAssemblyUnwindPlan(Target &target,
                                                          Thread &thread);

  std::shared_ptr<const UnwindPlan> GetObjectFileUnwindPlan(Target &target);

  std::shared_ptr<const UnwindPlan>
  GetObjectFileAugmentedUnwindPlan(Target &target, Thread &thread);

  std::shared_ptr<const UnwindPlan> GetEHFrameUnwindPlan(Target &target);

  std::shared_ptr<const UnwindPlan>
  GetEHFrameAugmentedUnwindPlan(Target &target, Thread &thread);

  std::shared_ptr<const UnwindPlan> GetDebugFrameUnwindPlan(Target &target);

  std::shared_ptr<const UnwindPlan>
  GetDebugFrameAugmentedUnwindPlan(Target &target, Thread &thread);

  std::shared_ptr<const UnwindPlan> GetCompactUnwindUnwindPlan(Target &target);

  std::shared_ptr<const UnwindPlan> GetArmUnwindUnwindPlan(Target &target);

  std::shared_ptr<const UnwindPlan> GetSymbolFileUnwindPlan(Thread &thread);

  std::shared_ptr<const UnwindPlan> GetArchDefaultUnwindPlan(Thread &thread);

  std::shared_ptr<const UnwindPlan>
  GetArchDefaultAtFuncEntryUnwindPlan(Thread &thread);

private:
  lldb::UnwindAssemblySP GetUnwindAssemblyProfiler(Target &target);

  // Do a simplistic comparison for the register restore rule for getting the
  // caller's pc value on two UnwindPlans -- returns LazyBoolYes if they have
  // the same unwind rule for the pc, LazyBoolNo if they do not have the same
  // unwind rule for the pc, and LazyBoolCalculate if it was unable to
  // determine this for some reason.
  lldb_private::LazyBool CompareUnwindPlansForIdenticalInitialPCLocation(
      Thread &thread, const std::shared_ptr<const UnwindPlan> &a,
      const std::shared_ptr<const UnwindPlan> &b);

  UnwindTable &m_unwind_table;

  /// Start address of the function described by this object.
  Address m_addr;

  /// The address ranges of the function.
  AddressRanges m_ranges;

  std::recursive_mutex m_mutex;

  std::shared_ptr<const UnwindPlan> m_unwind_plan_assembly_sp;
  std::shared_ptr<const UnwindPlan> m_unwind_plan_object_file_sp;
  std::shared_ptr<const UnwindPlan> m_unwind_plan_eh_frame_sp;
  std::shared_ptr<const UnwindPlan> m_unwind_plan_debug_frame_sp;

  // augmented by assembly inspection so it's valid everywhere
  std::shared_ptr<const UnwindPlan> m_unwind_plan_object_file_augmented_sp;
  std::shared_ptr<const UnwindPlan> m_unwind_plan_eh_frame_augmented_sp;
  std::shared_ptr<const UnwindPlan> m_unwind_plan_debug_frame_augmented_sp;

  std::vector<std::shared_ptr<const UnwindPlan>> m_unwind_plan_compact_unwind;
  std::shared_ptr<const UnwindPlan> m_unwind_plan_arm_unwind_sp;
  std::shared_ptr<const UnwindPlan> m_unwind_plan_symbol_file_sp;
  std::shared_ptr<const UnwindPlan> m_unwind_plan_fast_sp;
  std::shared_ptr<const UnwindPlan> m_unwind_plan_arch_default_sp;
  std::shared_ptr<const UnwindPlan> m_unwind_plan_arch_default_at_func_entry_sp;

  // Fetching the UnwindPlans can be expensive - if we've already attempted to
  // get one & failed, don't try again.
  bool m_tried_unwind_plan_assembly : 1, m_tried_unwind_plan_eh_frame : 1,
      m_tried_unwind_plan_object_file : 1,
      m_tried_unwind_plan_debug_frame : 1,
      m_tried_unwind_plan_object_file_augmented : 1,
      m_tried_unwind_plan_eh_frame_augmented : 1,
      m_tried_unwind_plan_debug_frame_augmented : 1,
      m_tried_unwind_plan_compact_unwind : 1,
      m_tried_unwind_plan_arm_unwind : 1, m_tried_unwind_plan_symbol_file : 1,
      m_tried_unwind_fast : 1, m_tried_unwind_arch_default : 1,
      m_tried_unwind_arch_default_at_func_entry : 1;

  Address m_first_non_prologue_insn;

  FuncUnwinders(const FuncUnwinders &) = delete;
  const FuncUnwinders &operator=(const FuncUnwinders &) = delete;

}; // class FuncUnwinders

} // namespace lldb_private

#endif // LLDB_SYMBOL_FUNCUNWINDERS_H
