#ifndef liblldb_FuncUnwinders_h
#define liblldb_FuncUnwinders_h


#include <memory>

#include "lldb/Core/AddressRange.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Host/Mutex.h"

namespace lldb_private {

class UnwindTable;

class FuncUnwinders 
{
public:
    // FuncUnwinders objects are used to track UnwindPlans for a function
    // (named or not - really just an address range)

    // We'll record three different UnwindPlans for each address range:
    //   1. Unwinding from a call site (a valid exception throw location)
    //      This is often sourced from the eh_frame exception handling info
    //   2. Unwinding from a non-call site (any location in the function)
    //      This is often done by analyzing the function prologue assembly
    //      langauge instructions
    //   3. A fast unwind method for this function which only retrieves a 
    //      limited set of registers necessary to walk the stack
    //   4. An architectural default unwind plan when none of the above are
    //      available for some reason.

    // Additionally, FuncUnwinds object can be asked where the prologue 
    // instructions are finished for migrating breakpoints past the 
    // stack frame setup instructions when we don't have line table information.

    FuncUnwinders (lldb_private::UnwindTable& unwind_table, lldb_private::UnwindAssembly *assembly_profiler, AddressRange range);

    ~FuncUnwinders ();

    // current_offset is the byte offset into the function.
    // 0 means no instructions have executed yet.  -1 means the offset is unknown.
    // On architectures where the pc points to the next instruction that will execute, this
    // offset value will have already been decremented by 1 to stay within the bounds of the 
    // correct function body.
    lldb::UnwindPlanSP
    GetUnwindPlanAtCallSite (int current_offset);

    lldb::UnwindPlanSP
    GetUnwindPlanAtNonCallSite (lldb_private::Thread& thread);

    lldb::UnwindPlanSP
    GetUnwindPlanFastUnwind (lldb_private::Thread& Thread);

    lldb::UnwindPlanSP
    GetUnwindPlanArchitectureDefault (lldb_private::Thread& thread);

    lldb::UnwindPlanSP
    GetUnwindPlanArchitectureDefaultAtFunctionEntry (lldb_private::Thread& thread);

    Address&
    GetFirstNonPrologueInsn (Target& target);

    const Address&
    GetFunctionStartAddress () const;

    bool
    ContainsAddress (const Address& addr) const
    { 
        return m_range.ContainsFileAddress (addr);
    }

    // When we're doing an unwind using the UnwindPlanAtNonCallSite and we find an
    // impossible unwind condition, we know that the UnwindPlan is invalid.  Calling
    // this method on the FuncUnwinder will tell it to replace that UnwindPlan with
    // the architectural default UnwindPlan so hopefully our stack walk will get past
    // this frame.
    void
    InvalidateNonCallSiteUnwindPlan (lldb_private::Thread& Thread);

private:
    UnwindTable& m_unwind_table;
    UnwindAssembly *m_assembly_profiler;
    AddressRange m_range;

    Mutex m_mutex;
    lldb::UnwindPlanSP m_unwind_plan_call_site_sp;
    lldb::UnwindPlanSP m_unwind_plan_non_call_site_sp;
    lldb::UnwindPlanSP m_unwind_plan_fast_sp;
    lldb::UnwindPlanSP m_unwind_plan_arch_default_sp;
    lldb::UnwindPlanSP m_unwind_plan_arch_default_at_func_entry_sp;

    bool m_tried_unwind_at_call_site:1,
         m_tried_unwind_at_non_call_site:1,
         m_tried_unwind_fast:1,
         m_tried_unwind_arch_default:1,
         m_tried_unwind_arch_default_at_func_entry:1;
         
         
    Address m_first_non_prologue_insn;

    DISALLOW_COPY_AND_ASSIGN (FuncUnwinders);

}; // class FuncUnwinders

} // namespace lldb_private


#endif //liblldb_FuncUnwinders_h
