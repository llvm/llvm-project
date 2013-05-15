//===-- Disassembler.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Disassembler_h_
#define liblldb_Disassembler_h_

// C Includes
// C++ Includes
#include <vector>
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/EmulateInstruction.h"
#include "lldb/Core/Opcode.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Interpreter/OptionValue.h"

namespace lldb_private {

class Instruction
{
public:
    Instruction (const Address &address, 
                 lldb::AddressClass addr_class = lldb::eAddressClassInvalid);

    virtual
   ~Instruction();

    const Address &
    GetAddress () const
    {
        return m_address;
    }
    
    const char *
    GetMnemonic (const ExecutionContext* exe_ctx)
    {
        CalculateMnemonicOperandsAndCommentIfNeeded (exe_ctx);
        return m_opcode_name.c_str();
    }
    const char *
    GetOperands (const ExecutionContext* exe_ctx)
    {
        CalculateMnemonicOperandsAndCommentIfNeeded (exe_ctx);
        return m_mnemonics.c_str();
    }
    
    const char *
    GetComment (const ExecutionContext* exe_ctx)
    {
        CalculateMnemonicOperandsAndCommentIfNeeded (exe_ctx);
        return m_comment.c_str();
    }

    virtual void
    CalculateMnemonicOperandsAndComment (const ExecutionContext* exe_ctx) = 0;
    
    lldb::AddressClass
    GetAddressClass ();

    void
    SetAddress (const Address &addr)
    {
        // Invalidate the address class to lazily discover
        // it if we need to.
        m_address_class = lldb::eAddressClassInvalid; 
        m_address = addr;
    }

    virtual void
    Dump (Stream *s,
          uint32_t max_opcode_byte_size,
          bool show_address,
          bool show_bytes,
          const ExecutionContext* exe_ctx);
    
    virtual bool
    DoesBranch () = 0;

    virtual size_t
    Decode (const Disassembler &disassembler, 
            const DataExtractor& data,
            lldb::offset_t data_offset) = 0;
            
    virtual void
    SetDescription (const char *) {}  // May be overridden in sub-classes that have descriptions.
    
    lldb::OptionValueSP
    ReadArray (FILE *in_file, Stream *out_stream, OptionValue::Type data_type);

    lldb::OptionValueSP
    ReadDictionary (FILE *in_file, Stream *out_stream);

    bool
    DumpEmulation (const ArchSpec &arch);
    
    virtual bool
    TestEmulation (Stream *stream, const char *test_file_name);
    
    bool
    Emulate (const ArchSpec &arch,
             uint32_t evaluate_options,
             void *baton,
             EmulateInstruction::ReadMemoryCallback read_mem_callback,
             EmulateInstruction::WriteMemoryCallback write_mem_calback,
             EmulateInstruction::ReadRegisterCallback read_reg_callback,
             EmulateInstruction::WriteRegisterCallback write_reg_callback);
                      
    const Opcode &
    GetOpcode () const
    {
        return m_opcode;
    }
    
    uint32_t
    GetData (DataExtractor &data);

protected:
    Address m_address; // The section offset address of this instruction
    // We include an address class in the Instruction class to
    // allow the instruction specify the eAddressClassCodeAlternateISA
    // (currently used for thumb), and also to specify data (eAddressClassData).
    // The usual value will be eAddressClassCode, but often when
    // disassembling memory, you might run into data. This can
    // help us to disassemble appropriately.
private:
    lldb::AddressClass m_address_class; // Use GetAddressClass () accessor function!
protected:
    Opcode m_opcode; // The opcode for this instruction
    std::string m_opcode_name;
    std::string m_mnemonics;
    std::string m_comment;
    bool m_calculated_strings;

    void
    CalculateMnemonicOperandsAndCommentIfNeeded (const ExecutionContext* exe_ctx)
    {
        if (!m_calculated_strings)
        {
            m_calculated_strings = true;
            CalculateMnemonicOperandsAndComment(exe_ctx);
        }
    }
};


class InstructionList
{
public:
    InstructionList();
    ~InstructionList();

    size_t
    GetSize() const;
    
    uint32_t
    GetMaxOpcocdeByteSize () const;

    lldb::InstructionSP
    GetInstructionAtIndex (size_t idx) const;
    
    uint32_t
    GetIndexOfNextBranchInstruction(uint32_t start) const;
    
    uint32_t
    GetIndexOfInstructionAtLoadAddress (lldb::addr_t load_addr, Target &target);

    void
    Clear();

    void
    Append (lldb::InstructionSP &inst_sp);

    void
    Dump (Stream *s,
          bool show_address,
          bool show_bytes,
          const ExecutionContext* exe_ctx);

private:
    typedef std::vector<lldb::InstructionSP> collection;
    typedef collection::iterator iterator;
    typedef collection::const_iterator const_iterator;

    collection m_instructions;
};

class PseudoInstruction : 
    public Instruction
{
public:

    PseudoInstruction ();
    
     virtual
     ~PseudoInstruction ();
     
    virtual bool
    DoesBranch ();

    virtual void
    CalculateMnemonicOperandsAndComment (const ExecutionContext* exe_ctx)
    {
        // TODO: fill this in and put opcode name into Instruction::m_opcode_name,
        // mnemonic into Instruction::m_mnemonics, and any comment into 
        // Instruction::m_comment
    }
    
    virtual size_t
    Decode (const Disassembler &disassembler,
            const DataExtractor &data,
            lldb::offset_t data_offset);
            
    void
    SetOpcode (size_t opcode_size, void *opcode_data);
    
    virtual void
    SetDescription (const char *description);
    
protected:
    std::string m_description;
    
    DISALLOW_COPY_AND_ASSIGN (PseudoInstruction);
};

class Disassembler :
    public std::enable_shared_from_this<Disassembler>,
    public PluginInterface
{
public:

    enum
    {
        eOptionNone             = 0u,
        eOptionShowBytes        = (1u << 0),
        eOptionRawOuput         = (1u << 1),
        eOptionMarkPCSourceLine = (1u << 2), // Mark the source line that contains the current PC (mixed mode only)
        eOptionMarkPCAddress    = (1u << 3)  // Mark the disassembly line the contains the PC
    };

    // FindPlugin should be lax about the flavor string (it is too annoying to have various internal uses of the
    // disassembler fail because the global flavor string gets set wrong.  Instead, if you get a flavor string you
    // don't understand, use the default.  Folks who care to check can use the FlavorValidForArchSpec method on the
    // disassembler they got back.
    static lldb::DisassemblerSP
    FindPlugin (const ArchSpec &arch, const char *flavor, const char *plugin_name);
    
    // This version will use the value in the Target settings if flavor is NULL;
    static lldb::DisassemblerSP
    FindPluginForTarget(const lldb::TargetSP target_sp, const ArchSpec &arch, const char *flavor, const char *plugin_name);

    static lldb::DisassemblerSP
    DisassembleRange (const ArchSpec &arch,
                      const char *plugin_name,
                      const char *flavor,
                      const ExecutionContext &exe_ctx,
                      const AddressRange &disasm_range);
    
    static lldb::DisassemblerSP 
    DisassembleBytes (const ArchSpec &arch,
                      const char *plugin_name,
                      const char *flavor,
                      const Address &start,
                      const void *bytes,
                      size_t length,
                      uint32_t max_num_instructions,
                      bool data_from_file);

    static bool
    Disassemble (Debugger &debugger,
                 const ArchSpec &arch,
                 const char *plugin_name,
                 const char *flavor,
                 const ExecutionContext &exe_ctx,
                 const AddressRange &range,
                 uint32_t num_instructions,
                 uint32_t num_mixed_context_lines,
                 uint32_t options,
                 Stream &strm);

    static bool
    Disassemble (Debugger &debugger,
                 const ArchSpec &arch,
                 const char *plugin_name,
                 const char *flavor,
                 const ExecutionContext &exe_ctx,
                 const Address &start,
                 uint32_t num_instructions,
                 uint32_t num_mixed_context_lines,
                 uint32_t options,
                 Stream &strm);

    static size_t
    Disassemble (Debugger &debugger,
                 const ArchSpec &arch,
                 const char *plugin_name,
                 const char *flavor,
                 const ExecutionContext &exe_ctx,
                 SymbolContextList &sc_list,
                 uint32_t num_instructions,
                 uint32_t num_mixed_context_lines,
                 uint32_t options,
                 Stream &strm);
    
    static bool
    Disassemble (Debugger &debugger,
                 const ArchSpec &arch,
                 const char *plugin_name,
                 const char *flavor,
                 const ExecutionContext &exe_ctx,
                 const ConstString &name,
                 Module *module,
                 uint32_t num_instructions,
                 uint32_t num_mixed_context_lines,
                 uint32_t options,
                 Stream &strm);

    static bool
    Disassemble (Debugger &debugger,
                 const ArchSpec &arch,
                 const char *plugin_name,
                 const char *flavor,
                 const ExecutionContext &exe_ctx,
                 uint32_t num_instructions,
                 uint32_t num_mixed_context_lines,
                 uint32_t options,
                 Stream &strm);
    
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    Disassembler(const ArchSpec &arch, const char *flavor);
    virtual ~Disassembler();

    typedef const char * (*SummaryCallback)(const Instruction& inst, ExecutionContext *exe_context, void *user_data);

    static bool 
    PrintInstructions (Disassembler *disasm_ptr,
                       Debugger &debugger,
                       const ArchSpec &arch,
                       const ExecutionContext &exe_ctx,
                       uint32_t num_instructions,
                       uint32_t num_mixed_context_lines,
                       uint32_t options,
                       Stream &strm);
    
    size_t
    ParseInstructions (const ExecutionContext *exe_ctx,
                       const AddressRange &range,
                       Stream *error_strm_ptr,
                       bool prefer_file_cache);

    size_t
    ParseInstructions (const ExecutionContext *exe_ctx,
                       const Address &range,
                       uint32_t num_instructions,
                       bool prefer_file_cache);

    virtual size_t
    DecodeInstructions (const Address &base_addr,
                        const DataExtractor& data,
                        lldb::offset_t data_offset,
                        size_t num_instructions,
                        bool append,
                        bool data_from_file) = 0;
    
    InstructionList &
    GetInstructionList ();

    const InstructionList &
    GetInstructionList () const;

    const ArchSpec &
    GetArchitecture () const
    {
        return m_arch;
    }
    
    const char *
    GetFlavor () const
    {
        return m_flavor.c_str();
    }
    
    virtual bool
    FlavorValidForArchSpec (const lldb_private::ArchSpec &arch, const char *flavor) = 0;    

protected:
    //------------------------------------------------------------------
    // Classes that inherit from Disassembler can see and modify these
    //------------------------------------------------------------------
    const ArchSpec m_arch;
    InstructionList m_instruction_list;
    lldb::addr_t m_base_addr;
    std::string m_flavor;

private:
    //------------------------------------------------------------------
    // For Disassembler only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (Disassembler);
};

} // namespace lldb_private

#endif  // liblldb_Disassembler_h_
