//===-- RecordingMemoryManager.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_RecordingMemoryManager_h_
#define lldb_RecordingMemoryManager_h_

// C Includes
// C++ Includes
#include <string>
#include <vector>
#include <map>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/Log.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangExpressionParser.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class RecordingMemoryManager RecordingMemoryManager.h "lldb/Expression/RecordingMemoryManager.h"
/// @brief Passthrough memory manager for the JIT that records what was allocated where
///
/// The LLVM JIT is built to compile code for execution in the current
/// process, so it needs to be able to allocate memory.  Because different
/// clients have different requirements for the locations of JIT compiled
/// code, the interface for allocating memory has been abstracted out and
/// can be implemented by any client.
///
/// LLDB, however, needs to move JIT-compiled code into the target process.
/// Because writing individual bytes of code hasn't been abstracted out of
/// the JIT, LLDB instead implements a custom memory allocator that records 
/// what regions have been allocated for code.  When JIT compilation is
/// complete, these regions are then copied as necessary into the target
/// process.
///
/// Ideally the memory manager would handle this copying, but this class has 
/// to be built without RTTI, which means it cannot include Process.h.  As a
/// result, ClangExpression::WriteJITCode() accesses the stored mappings 
/// directly.
//----------------------------------------------------------------------
class RecordingMemoryManager : public llvm::JITMemoryManager
{
public:
    //------------------------------------------------------------------
    /// Constructor
    //------------------------------------------------------------------
    RecordingMemoryManager ();
    
    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    virtual ~RecordingMemoryManager();

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual void setMemoryWritable ();

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual void setMemoryExecutable ();

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual void setPoisonMemory (bool poison)
    {
        m_default_mm_ap->setPoisonMemory (poison);
    }

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual void AllocateGOT()
    {
        m_default_mm_ap->AllocateGOT();
    }

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual uint8_t *getGOTBase() const
    {
        return m_default_mm_ap->getGOTBase();
    }

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual uint8_t *startFunctionBody(const llvm::Function *F,
                                       uintptr_t &ActualSize);

    //------------------------------------------------------------------
    /// Allocate room for a dyld stub for a lazy-referenced function,
    /// and add it to the m_stubs map
    ///
    /// @param[in] F
    ///     The function being referenced.
    ///
    /// @param[in] StubSize
    ///     The size of the stub.
    ///
    /// @param[in] Alignment
    ///     The required alignment of the stub.
    ///
    /// @return
    ///     Allocated space for the stub.
    //------------------------------------------------------------------
    virtual uint8_t *allocateStub(const llvm::GlobalValue* F, 
                                  unsigned StubSize,
                                  unsigned Alignment);

    //------------------------------------------------------------------
    /// Complete the body of a function, and add it to the m_functions map
    ///
    /// @param[in] F
    ///     The function being completed.
    ///
    /// @param[in] FunctionStart
    ///     The first instruction of the function.
    ///
    /// @param[in] FunctionEnd
    ///     The last byte of the last instruction of the function.
    //------------------------------------------------------------------
    virtual void endFunctionBody(const llvm::Function *F, 
                                 uint8_t *FunctionStart,
                                 uint8_t *FunctionEnd);
    //------------------------------------------------------------------
    /// Allocate space for an unspecified purpose, and add it to the
    /// m_spaceBlocks map
    ///
    /// @param[in] Size
    ///     The size of the area.
    ///
    /// @param[in] Alignment
    ///     The required alignment of the area.
    ///
    /// @return
    ///     Allocated space.
    //------------------------------------------------------------------
    virtual uint8_t *allocateSpace(intptr_t Size, unsigned Alignment);
    
    //------------------------------------------------------------------
    /// Allocate space for executable code, and add it to the 
    /// m_spaceBlocks map
    ///
    /// @param[in] Size
    ///     The size of the area.
    ///
    /// @param[in] Alignment
    ///     The required alignment of the area.
    ///
    /// @param[in] SectionID
    ///     A unique identifier for the section.
    ///
    /// @return
    ///     Allocated space.
    //------------------------------------------------------------------
    virtual uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                         unsigned SectionID);
    
    //------------------------------------------------------------------
    /// Allocate space for data, and add it to the m_spaceBlocks map
    ///
    /// @param[in] Size
    ///     The size of the area.
    ///
    /// @param[in] Alignment
    ///     The required alignment of the area.
    ///
    /// @param[in] SectionID
    ///     A unique identifier for the section.
    ///
    /// @return
    ///     Allocated space.
    //------------------------------------------------------------------
    virtual uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                         unsigned SectionID);

    //------------------------------------------------------------------
    /// Allocate space for a global variable, and add it to the
    /// m_spaceBlocks map
    ///
    /// @param[in] Size
    ///     The size of the variable.
    ///
    /// @param[in] Alignment
    ///     The required alignment of the variable.
    ///
    /// @return
    ///     Allocated space for the global.
    //------------------------------------------------------------------
    virtual uint8_t *allocateGlobal(uintptr_t Size, 
                                    unsigned Alignment);

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual void deallocateFunctionBody(void *Body);

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual uint8_t* startExceptionTable(const llvm::Function* F,
                                         uintptr_t &ActualSize);

    //------------------------------------------------------------------
    /// Complete the exception table for a function, and add it to the
    /// m_exception_tables map
    ///
    /// @param[in] F
    ///     The function whose exception table is being written.
    ///
    /// @param[in] TableStart
    ///     The first byte of the exception table.
    ///
    /// @param[in] TableEnd
    ///     The last byte of the exception table.
    ///
    /// @param[in] FrameRegister
    ///     I don't know what this does, but it's passed through.
    //------------------------------------------------------------------
    virtual void endExceptionTable(const llvm::Function *F, 
                                   uint8_t *TableStart,
                                   uint8_t *TableEnd, 
                                   uint8_t* FrameRegister);

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual void deallocateExceptionTable(void *ET);

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual size_t GetDefaultCodeSlabSize() {
        return m_default_mm_ap->GetDefaultCodeSlabSize();
    }

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual size_t GetDefaultDataSlabSize() {
        return m_default_mm_ap->GetDefaultDataSlabSize();
    }

    virtual size_t GetDefaultStubSlabSize() {
        return m_default_mm_ap->GetDefaultStubSlabSize();
    }

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual unsigned GetNumCodeSlabs() {
        return m_default_mm_ap->GetNumCodeSlabs();
    }

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual unsigned GetNumDataSlabs() {
        return m_default_mm_ap->GetNumDataSlabs();
    }

    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual unsigned GetNumStubSlabs() {
        return m_default_mm_ap->GetNumStubSlabs();
    }
    
    //------------------------------------------------------------------
    /// Passthrough interface stub
    //------------------------------------------------------------------
    virtual void *getPointerToNamedFunction(const std::string &Name,
                                            bool AbortOnFailure = true) {
        return m_default_mm_ap->getPointerToNamedFunction(Name, AbortOnFailure);
    }

    //------------------------------------------------------------------
    /// [Convenience method for ClangExpressionParser] Look up the object in
    /// m_address_map that contains a given address, find where it was
    /// copied to, and return the remote address at the same offset into
    /// the copied entity
    ///
    /// @param[in] local_address
    ///     The address in the debugger.
    ///
    /// @return
    ///     The address in the target process.
    //------------------------------------------------------------------
    lldb::addr_t
    GetRemoteAddressForLocal (lldb::addr_t local_address);
    
    //------------------------------------------------------------------
    /// [Convenience method for ClangExpressionParser] Look up the object in
    /// m_address_map that contains a given address, find where it was
    /// copied to, and return its address range in the target process
    ///
    /// @param[in] local_address
    ///     The address in the debugger.
    ///
    /// @return
    ///     The range of the containing object in the target process.
    //------------------------------------------------------------------
    typedef std::pair <lldb::addr_t, uintptr_t> AddrRange;
    AddrRange
    GetRemoteRangeForLocal (lldb::addr_t local_address);
    
    //------------------------------------------------------------------
    /// [Convenience method for ClangExpressionParser] Commit all allocations
    /// to the process and record where they were stored.
    ///
    /// @param[in] process
    ///     The process to allocate memory in.
    ///
    /// @return
    ///     True <=> all allocations were performed successfully.
    ///     This method will attempt to free allocated memory if the
    ///     operation fails.
    //------------------------------------------------------------------
    bool
    CommitAllocations (Process &process);
    
    //------------------------------------------------------------------
    /// [Convenience method for ClangExpressionParser] Report all committed
    /// allocations to the execution engine.
    ///
    /// @param[in] engine
    ///     The execution engine to notify.
    //------------------------------------------------------------------
    void
    ReportAllocations (llvm::ExecutionEngine &engine);
    
    //------------------------------------------------------------------
    /// [Convenience method for ClangExpressionParser] Write the contents
    /// of all allocations to the process. 
    ///
    /// @param[in] local_address
    ///     The process containing the allocations.
    ///
    /// @return
    ///     True <=> all allocations were performed successfully.
    //------------------------------------------------------------------
    bool
    WriteData (Process &process);
private:
    std::auto_ptr<JITMemoryManager> m_default_mm_ap;    ///< The memory allocator to use in actually creating space.  All calls are passed through to it.
    
    lldb::LogSP m_log; ///< The log to use when printing log messages.  May be NULL.

    //----------------------------------------------------------------------
    /// @class Allocation RecordingMemoryManager.h "lldb/Expression/RecordingMemoryManager.h"
    /// @brief A record of a region that has been allocated by the JIT.
    ///
    /// The RecordingMemoryManager makes records of all regions that need copying;
    /// upon requests, it allocates and 
    //----------------------------------------------------------------------
    struct Allocation
    {
        lldb::addr_t    m_remote_allocation;///< The (unaligned) base for the remote allocation
        lldb::addr_t    m_remote_start;     ///< The base address of the remote allocation
        uintptr_t       m_local_start;      ///< The base address of the local allocation
        uintptr_t       m_size;             ///< The size of the allocation
        unsigned        m_section_id;       ///< The ID of the section
        unsigned        m_alignment;        ///< The required alignment for the allocation
        bool            m_executable;       ///< True <=> the allocation must be executable in the target
        bool            m_allocated;        ///< True <=> the allocation has been propagated to the target

        static const unsigned eSectionIDNone = (unsigned)-1;
        
        //------------------------------------------------------------------
        /// Constructor
        //------------------------------------------------------------------
        Allocation () :
            m_remote_allocation(0),
            m_remote_start(0),
            m_local_start(0),
            m_size(0),
            m_section_id(eSectionIDNone),
            m_alignment(0),
            m_executable(false),
            m_allocated(false)
        {
        }
        
        void dump (lldb::LogSP log);
    };

    typedef std::vector<Allocation> AllocationList;
    AllocationList                  m_allocations;  ///< The base address of the remote allocation
};

} // namespace lldb_private

#endif  // lldb_RecordingMemoryManager_h_
