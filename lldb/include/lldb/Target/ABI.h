//===-- ABI.h ---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ABI_h_
#define liblldb_ABI_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class ABI :
    public PluginInterface
{
public:
    virtual
    ~ABI();

    virtual size_t
    GetRedZoneSize () const = 0;

    virtual bool
    PrepareTrivialCall (Thread &thread, 
                        lldb::addr_t sp,
                        lldb::addr_t functionAddress,
                        lldb::addr_t returnAddress, 
                        lldb::addr_t *arg1_ptr = NULL,
                        lldb::addr_t *arg2_ptr = NULL,
                        lldb::addr_t *arg3_ptr = NULL,
                        lldb::addr_t *arg4_ptr = NULL,
                        lldb::addr_t *arg5_ptr = NULL,
                        lldb::addr_t *arg6_ptr = NULL) const = 0;

    virtual bool
    GetArgumentValues (Thread &thread,
                       ValueList &values) const = 0;
    
    lldb::ValueObjectSP
    GetReturnValueObject (Thread &thread,
                          ClangASTType &type,
                          bool persistent = true) const;
    
    // Set the Return value object in the current frame as though a function with 
    virtual Error
    SetReturnValueObject(lldb::StackFrameSP &frame_sp, lldb::ValueObjectSP &new_value) = 0;

protected:    
    // This is the method the ABI will call to actually calculate the return value.
    // Don't put it in a persistant value object, that will be done by the ABI::GetReturnValueObject.
    virtual lldb::ValueObjectSP
    GetReturnValueObjectImpl (Thread &thread,
                          ClangASTType &type) const = 0;
public:
    virtual bool
    CreateFunctionEntryUnwindPlan (UnwindPlan &unwind_plan) = 0;

    virtual bool
    CreateDefaultUnwindPlan (UnwindPlan &unwind_plan) = 0;

    virtual bool
    RegisterIsVolatile (const RegisterInfo *reg_info) = 0;

    // Should return true if your ABI uses frames when doing stack backtraces. This
    // means a frame pointer is used that points to the previous stack frame in some
    // way or another.
    virtual bool
    StackUsesFrames () = 0;

    // Should take a look at a call frame address (CFA) which is just the stack
    // pointer value upon entry to a function. ABIs usually impose alignment
    // restrictions (4, 8 or 16 byte aligned), and zero is usually not allowed.
    // This function should return true if "cfa" is valid call frame address for
    // the ABI, and false otherwise. This is used by the generic stack frame unwinding
    // code to help determine when a stack ends.
    virtual bool
    CallFrameAddressIsValid (lldb::addr_t cfa) = 0;

    // Validates a possible PC value and returns true if an opcode can be at "pc".
    virtual bool
    CodeAddressIsValid (lldb::addr_t pc) = 0;    

    virtual lldb::addr_t
    FixCodeAddress (lldb::addr_t pc)
    {
        // Some targets might use bits in a code address to indicate
        // a mode switch. ARM uses bit zero to signify a code address is
        // thumb, so any ARM ABI plug-ins would strip those bits.
        return pc;
    }

    virtual const RegisterInfo *
    GetRegisterInfoArray (uint32_t &count) = 0;

    // Some architectures (e.g. x86) will push the return address on the stack and decrement
    // the stack pointer when making a function call.  This means that every stack frame will
    // have a unique CFA.
    // Other architectures (e.g. arm) pass the return address in a register so it is possible
    // to have a frame on a backtrace that does not push anything on the stack or change the 
    // CFA.
    virtual bool
    FunctionCallsChangeCFA () = 0;

    
    bool
    GetRegisterInfoByName (const ConstString &name, RegisterInfo &info);

    bool
    GetRegisterInfoByKind (lldb::RegisterKind reg_kind, 
                           uint32_t reg_num, 
                           RegisterInfo &info);

    static lldb::ABISP
    FindPlugin (const ArchSpec &arch);
    
protected:
    //------------------------------------------------------------------
    // Classes that inherit from ABI can see and modify these
    //------------------------------------------------------------------
    ABI();
private:
    DISALLOW_COPY_AND_ASSIGN (ABI);
};

} // namespace lldb_private

#endif  // liblldb_ABI_h_
