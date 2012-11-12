//===-- Disassembler.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Disassembler.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/EmulateInstruction.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Timer.h"
#include "lldb/Interpreter/OptionValue.h"
#include "lldb/Interpreter/OptionValueArray.h"
#include "lldb/Interpreter/OptionValueDictionary.h"
#include "lldb/Interpreter/OptionValueString.h"
#include "lldb/Interpreter/OptionValueUInt64.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"

#define DEFAULT_DISASM_BYTE_SIZE 32

using namespace lldb;
using namespace lldb_private;


DisassemblerSP
Disassembler::FindPlugin (const ArchSpec &arch, const char *plugin_name)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "Disassembler::FindPlugin (arch = %s, plugin_name = %s)",
                        arch.GetArchitectureName(),
                        plugin_name);

    DisassemblerCreateInstance create_callback = NULL;
    
    if (plugin_name)
    {
        create_callback = PluginManager::GetDisassemblerCreateCallbackForPluginName (plugin_name);
        if (create_callback)
        {
            DisassemblerSP disassembler_sp(create_callback(arch));
            
            if (disassembler_sp.get())
                return disassembler_sp;
        }
    }
    else
    {
        for (uint32_t idx = 0; (create_callback = PluginManager::GetDisassemblerCreateCallbackAtIndex(idx)) != NULL; ++idx)
        {
            DisassemblerSP disassembler_sp(create_callback(arch));

            if (disassembler_sp.get())
                return disassembler_sp;
        }
    }
    return DisassemblerSP();
}


static void
ResolveAddress (const ExecutionContext &exe_ctx,
                const Address &addr, 
                Address &resolved_addr)
{
    if (!addr.IsSectionOffset())
    {
        // If we weren't passed in a section offset address range,
        // try and resolve it to something
        Target *target = exe_ctx.GetTargetPtr();
        if (target)
        {
            if (target->GetSectionLoadList().IsEmpty())
            {
                target->GetImages().ResolveFileAddress (addr.GetOffset(), resolved_addr);
            }
            else
            {
                target->GetSectionLoadList().ResolveLoadAddress (addr.GetOffset(), resolved_addr);
            }
            // We weren't able to resolve the address, just treat it as a
            // raw address
            if (resolved_addr.IsValid())
                return;
        }
    }
    resolved_addr = addr;
}

size_t
Disassembler::Disassemble
(
    Debugger &debugger,
    const ArchSpec &arch,
    const char *plugin_name,
    const ExecutionContext &exe_ctx,
    SymbolContextList &sc_list,
    uint32_t num_instructions,
    uint32_t num_mixed_context_lines,
    uint32_t options,
    Stream &strm
)
{
    size_t success_count = 0;
    const size_t count = sc_list.GetSize();
    SymbolContext sc;
    AddressRange range;
    const uint32_t scope = eSymbolContextBlock | eSymbolContextFunction | eSymbolContextSymbol;
    const bool use_inline_block_range = true;
    for (size_t i=0; i<count; ++i)
    {
        if (sc_list.GetContextAtIndex(i, sc) == false)
            break;
        for (uint32_t range_idx = 0; sc.GetAddressRange(scope, range_idx, use_inline_block_range, range); ++range_idx)
        {
            if (Disassemble (debugger, 
                             arch, 
                             plugin_name,
                             exe_ctx, 
                             range, 
                             num_instructions,
                             num_mixed_context_lines, 
                             options, 
                             strm))
            {
                ++success_count;
                strm.EOL();
            }
        }
    }
    return success_count;
}

bool
Disassembler::Disassemble
(
    Debugger &debugger,
    const ArchSpec &arch,
    const char *plugin_name,
    const ExecutionContext &exe_ctx,
    const ConstString &name,
    Module *module,
    uint32_t num_instructions,
    uint32_t num_mixed_context_lines,
    uint32_t options,
    Stream &strm
)
{
    SymbolContextList sc_list;
    if (name)
    {
        const bool include_symbols = true;
        const bool include_inlines = true;
        if (module)
        {
            module->FindFunctions (name,
                                   NULL,
                                   eFunctionNameTypeBase | 
                                   eFunctionNameTypeFull | 
                                   eFunctionNameTypeMethod | 
                                   eFunctionNameTypeSelector, 
                                   include_symbols,
                                   include_inlines,
                                   true,
                                   sc_list);
        }
        else if (exe_ctx.GetTargetPtr())
        {
            exe_ctx.GetTargetPtr()->GetImages().FindFunctions (name, 
                                                               eFunctionNameTypeBase | 
                                                               eFunctionNameTypeFull | 
                                                               eFunctionNameTypeMethod | 
                                                               eFunctionNameTypeSelector,
                                                               include_symbols,
                                                               include_inlines,
                                                               false,
                                                               sc_list);
        }
    }
    
    if (sc_list.GetSize ())
    {
        return Disassemble (debugger, 
                            arch, 
                            plugin_name,
                            exe_ctx, 
                            sc_list,
                            num_instructions, 
                            num_mixed_context_lines, 
                            options,
                            strm);
    }
    return false;
}


lldb::DisassemblerSP
Disassembler::DisassembleRange
(
    const ArchSpec &arch,
    const char *plugin_name,
    const ExecutionContext &exe_ctx,
    const AddressRange &range
)
{
    lldb::DisassemblerSP disasm_sp;
    if (range.GetByteSize() > 0 && range.GetBaseAddress().IsValid())
    {
        disasm_sp = Disassembler::FindPlugin(arch, plugin_name);

        if (disasm_sp)
        {
            size_t bytes_disassembled = disasm_sp->ParseInstructions (&exe_ctx, range, NULL);
            if (bytes_disassembled == 0)
                disasm_sp.reset();
        }
    }
    return disasm_sp;
}

lldb::DisassemblerSP 
Disassembler::DisassembleBytes 
(
    const ArchSpec &arch,
    const char *plugin_name,
    const Address &start,
    const void *bytes,
    size_t length,
    uint32_t num_instructions
)
{
    lldb::DisassemblerSP disasm_sp;
    
    if (bytes)
    {
        disasm_sp = Disassembler::FindPlugin(arch, plugin_name);
        
        if (disasm_sp)
        {
            DataExtractor data(bytes, length, arch.GetByteOrder(), arch.GetAddressByteSize());
            
            (void)disasm_sp->DecodeInstructions (start,
                                                 data,
                                                 0,
                                                 num_instructions,
                                                 false);
        }
    }
    
    return disasm_sp;
}


bool
Disassembler::Disassemble
(
    Debugger &debugger,
    const ArchSpec &arch,
    const char *plugin_name,
    const ExecutionContext &exe_ctx,
    const AddressRange &disasm_range,
    uint32_t num_instructions,
    uint32_t num_mixed_context_lines,
    uint32_t options,
    Stream &strm
)
{
    if (disasm_range.GetByteSize())
    {
        lldb::DisassemblerSP disasm_sp (Disassembler::FindPlugin(arch, plugin_name));

        if (disasm_sp.get())
        {
            AddressRange range;
            ResolveAddress (exe_ctx, disasm_range.GetBaseAddress(), range.GetBaseAddress());
            range.SetByteSize (disasm_range.GetByteSize());
            
            size_t bytes_disassembled = disasm_sp->ParseInstructions (&exe_ctx, range, &strm);
            if (bytes_disassembled == 0)
                return false;

            return PrintInstructions (disasm_sp.get(),
                                      debugger,
                                      arch,
                                      exe_ctx,
                                      num_instructions,
                                      num_mixed_context_lines,
                                      options,
                                      strm);
        }
    }
    return false;
}
            
bool
Disassembler::Disassemble
(
    Debugger &debugger,
    const ArchSpec &arch,
    const char *plugin_name,
    const ExecutionContext &exe_ctx,
    const Address &start_address,
    uint32_t num_instructions,
    uint32_t num_mixed_context_lines,
    uint32_t options,
    Stream &strm
)
{
    if (num_instructions > 0)
    {
        lldb::DisassemblerSP disasm_sp (Disassembler::FindPlugin(arch, plugin_name));
        if (disasm_sp.get())
        {
            Address addr;
            ResolveAddress (exe_ctx, start_address, addr);

            size_t bytes_disassembled = disasm_sp->ParseInstructions (&exe_ctx, addr, num_instructions);
            if (bytes_disassembled == 0)
                return false;
            return PrintInstructions (disasm_sp.get(),
                                      debugger,
                                      arch,
                                      exe_ctx,
                                      num_instructions,
                                      num_mixed_context_lines,
                                      options,
                                      strm);
        }
    }
    return false;
}
            
bool 
Disassembler::PrintInstructions
(
    Disassembler *disasm_ptr,
    Debugger &debugger,
    const ArchSpec &arch,
    const ExecutionContext &exe_ctx,
    uint32_t num_instructions,
    uint32_t num_mixed_context_lines,
    uint32_t options,
    Stream &strm
)
{
    // We got some things disassembled...
    size_t num_instructions_found = disasm_ptr->GetInstructionList().GetSize();
    
    if (num_instructions > 0 && num_instructions < num_instructions_found)
        num_instructions_found = num_instructions;
        
    const uint32_t max_opcode_byte_size = disasm_ptr->GetInstructionList().GetMaxOpcocdeByteSize ();
    uint32_t offset = 0;
    SymbolContext sc;
    SymbolContext prev_sc;
    AddressRange sc_range;
    const Address *pc_addr_ptr = NULL;
    ExecutionContextScope *exe_scope = exe_ctx.GetBestExecutionContextScope();
    StackFrame *frame = exe_ctx.GetFramePtr();

    if (frame)
        pc_addr_ptr = &frame->GetFrameCodeAddress();
    const uint32_t scope = eSymbolContextLineEntry | eSymbolContextFunction | eSymbolContextSymbol;
    const bool use_inline_block_range = false;
    for (size_t i=0; i<num_instructions_found; ++i)
    {
        Instruction *inst = disasm_ptr->GetInstructionList().GetInstructionAtIndex (i).get();
        if (inst)
        {
            const Address &addr = inst->GetAddress();
            const bool inst_is_at_pc = pc_addr_ptr && addr == *pc_addr_ptr;

            prev_sc = sc;

            ModuleSP module_sp (addr.GetModule());
            if (module_sp)
            {
                uint32_t resolved_mask = module_sp->ResolveSymbolContextForAddress(addr, eSymbolContextEverything, sc);
                if (resolved_mask)
                {
                    if (num_mixed_context_lines)
                    {
                        if (!sc_range.ContainsFileAddress (addr))
                        {
                            sc.GetAddressRange (scope, 0, use_inline_block_range, sc_range);
                            
                            if (sc != prev_sc)
                            {
                                if (offset != 0)
                                    strm.EOL();
                                
                                sc.DumpStopContext(&strm, exe_ctx.GetProcessPtr(), addr, false, true, false);
                                strm.EOL();
                                
                                if (sc.comp_unit && sc.line_entry.IsValid())
                                {
                                    debugger.GetSourceManager().DisplaySourceLinesWithLineNumbers (sc.line_entry.file,
                                                                                                   sc.line_entry.line,
                                                                                                   num_mixed_context_lines,
                                                                                                   num_mixed_context_lines,
                                                                                                   ((inst_is_at_pc && (options & eOptionMarkPCSourceLine)) ? "->" : ""),
                                                                                                   &strm);
                                }
                            }
                        }
                    }
                    else if ((sc.function || sc.symbol) && (sc.function != prev_sc.function || sc.symbol != prev_sc.symbol))
                    {
                        if (prev_sc.function || prev_sc.symbol)
                            strm.EOL();

                        bool show_fullpaths = false;
                        bool show_module = true;
                        bool show_inlined_frames = true;
                        sc.DumpStopContext (&strm, 
                                            exe_scope, 
                                            addr, 
                                            show_fullpaths,
                                            show_module,
                                            show_inlined_frames);
                        
                        strm << ":\n";
                    }
                }
                else
                {
                    sc.Clear();
                }
            }

            if ((options & eOptionMarkPCAddress) && pc_addr_ptr)
            {
                strm.PutCString(inst_is_at_pc ? "-> " : "   ");
            }
            const bool show_bytes = (options & eOptionShowBytes) != 0;
            inst->Dump(&strm, max_opcode_byte_size, true, show_bytes, &exe_ctx);
            strm.EOL();            
        }
        else
        {
            break;
        }
    }
        
    return true;
}


bool
Disassembler::Disassemble
(
    Debugger &debugger,
    const ArchSpec &arch,
    const char *plugin_name,
    const ExecutionContext &exe_ctx,
    uint32_t num_instructions,
    uint32_t num_mixed_context_lines,
    uint32_t options,
    Stream &strm
)
{
    AddressRange range;
    StackFrame *frame = exe_ctx.GetFramePtr();
    if (frame)
    {
        SymbolContext sc(frame->GetSymbolContext(eSymbolContextFunction | eSymbolContextSymbol));
        if (sc.function)
        {
            range = sc.function->GetAddressRange();
        }
        else if (sc.symbol && sc.symbol->ValueIsAddress())
        {
            range.GetBaseAddress() = sc.symbol->GetAddress();
            range.SetByteSize (sc.symbol->GetByteSize());
        }
        else
        {
            range.GetBaseAddress() = frame->GetFrameCodeAddress();
        }

        if (range.GetBaseAddress().IsValid() && range.GetByteSize() == 0)
            range.SetByteSize (DEFAULT_DISASM_BYTE_SIZE);
    }

    return Disassemble (debugger, 
                        arch, 
                        plugin_name,
                        exe_ctx, 
                        range, 
                        num_instructions, 
                        num_mixed_context_lines, 
                        options, 
                        strm);
}

Instruction::Instruction(const Address &address, AddressClass addr_class) :
    m_address (address),
    m_address_class (addr_class),
    m_opcode(),
    m_calculated_strings(false)
{
}

Instruction::~Instruction()
{
}

AddressClass
Instruction::GetAddressClass ()
{
    if (m_address_class == eAddressClassInvalid)
        m_address_class = m_address.GetAddressClass();
    return m_address_class;
}

void
Instruction::Dump (lldb_private::Stream *s,
                   uint32_t max_opcode_byte_size,
                   bool show_address,
                   bool show_bytes,
                   const ExecutionContext* exe_ctx)
{
    const size_t opcode_column_width = 7;
    const size_t operand_column_width = 25;
    
    CalculateMnemonicOperandsAndCommentIfNeeded (exe_ctx);

    StreamString ss;
    
    if (show_address)
    {
        m_address.Dump(&ss,
                       exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL,
                       Address::DumpStyleLoadAddress,
                       Address::DumpStyleModuleWithFileAddress,
                       0);
        
        ss.PutCString(":  ");
    }
    
    if (show_bytes)
    {
        if (m_opcode.GetType() == Opcode::eTypeBytes)
        {
            // x86_64 and i386 are the only ones that use bytes right now so
            // pad out the byte dump to be able to always show 15 bytes (3 chars each) 
            // plus a space
            if (max_opcode_byte_size > 0)
                m_opcode.Dump (&ss, max_opcode_byte_size * 3 + 1);
            else
                m_opcode.Dump (&ss, 15 * 3 + 1);
        }
        else
        {
            // Else, we have ARM which can show up to a uint32_t 0x00000000 (10 spaces)
            // plus two for padding...
            if (max_opcode_byte_size > 0)
                m_opcode.Dump (&ss, max_opcode_byte_size * 3 + 1);
            else
                m_opcode.Dump (&ss, 12);
        }        
    }
    
    const size_t opcode_pos = ss.GetSize();
    
    ss.PutCString (m_opcode_name.c_str());
    ss.FillLastLineToColumn (opcode_pos + opcode_column_width, ' ');
    ss.PutCString (m_mnemocics.c_str());
    
    if (!m_comment.empty())
    {
        ss.FillLastLineToColumn (opcode_pos + opcode_column_width + operand_column_width, ' ');        
        ss.PutCString (" ; ");
        ss.PutCString (m_comment.c_str());
    }
    s->Write (ss.GetData(), ss.GetSize());
}

bool
Instruction::DumpEmulation (const ArchSpec &arch)
{
	std::auto_ptr<EmulateInstruction> insn_emulator_ap (EmulateInstruction::FindPlugin (arch, eInstructionTypeAny, NULL));
	if (insn_emulator_ap.get())
	{
        insn_emulator_ap->SetInstruction (GetOpcode(), GetAddress(), NULL);
        return insn_emulator_ap->EvaluateInstruction (0);
	}

    return false;
}

OptionValueSP
Instruction::ReadArray (FILE *in_file, Stream *out_stream, OptionValue::Type data_type)
{
    bool done = false;
    char buffer[1024];
    
    OptionValueSP option_value_sp (new OptionValueArray (1u << data_type));
    
    int idx = 0;
    while (!done)
    {
        if (!fgets (buffer, 1023, in_file))
        {
            out_stream->Printf ("Instruction::ReadArray:  Error reading file (fgets).\n");
            option_value_sp.reset ();
            return option_value_sp;
        }

        std::string line (buffer);
        
        int len = line.size();
        if (line[len-1] == '\n')
        {
            line[len-1] = '\0';
            line.resize (len-1);
        }

        if ((line.size() == 1) && line[0] == ']')
        {
            done = true;
            line.clear();
        }

        if (line.size() > 0)
        {
            std::string value;
            RegularExpression reg_exp ("^[ \t]*([^ \t]+)[ \t]*$");
            bool reg_exp_success = reg_exp.Execute (line.c_str(), 1);
            if (reg_exp_success)
                reg_exp.GetMatchAtIndex (line.c_str(), 1, value);
            else
                value = line;
                
            OptionValueSP data_value_sp;
            switch (data_type)
            {
            case OptionValue::eTypeUInt64:
                data_value_sp.reset (new OptionValueUInt64 (0, 0));
                data_value_sp->SetValueFromCString (value.c_str());
                break;
            // Other types can be added later as needed.
            default:
                data_value_sp.reset (new OptionValueString (value.c_str(), ""));
                break;
            }

            option_value_sp->GetAsArray()->InsertValue (idx, data_value_sp);
            ++idx;
        }
    }
    
    return option_value_sp;
}

OptionValueSP 
Instruction::ReadDictionary (FILE *in_file, Stream *out_stream)
{
    bool done = false;
    char buffer[1024];
    
    OptionValueSP option_value_sp (new OptionValueDictionary());
    static ConstString encoding_key ("data_encoding");
    OptionValue::Type data_type = OptionValue::eTypeInvalid;

    
    while (!done)
    {
        // Read the next line in the file
        if (!fgets (buffer, 1023, in_file))
        {
            out_stream->Printf ("Instruction::ReadDictionary: Error reading file (fgets).\n");
            option_value_sp.reset ();
            return option_value_sp;
        }
        
        // Check to see if the line contains the end-of-dictionary marker ("}")
        std::string line (buffer);

        int len = line.size();
        if (line[len-1] == '\n')
        {
            line[len-1] = '\0';
            line.resize (len-1);
        }
        
        if ((line.size() == 1) && (line[0] == '}'))
        {
            done = true;
            line.clear();
        }
        
        // Try to find a key-value pair in the current line and add it to the dictionary.
        if (line.size() > 0)
        {
            RegularExpression reg_exp ("^[ \t]*([a-zA-Z_][a-zA-Z0-9_]*)[ \t]*=[ \t]*(.*)[ \t]*$");
            bool reg_exp_success = reg_exp.Execute (line.c_str(), 2);
            std::string key;
            std::string value;
            if (reg_exp_success)
            {
                reg_exp.GetMatchAtIndex (line.c_str(), 1, key);
                reg_exp.GetMatchAtIndex (line.c_str(), 2, value);
            }
            else 
            {
                out_stream->Printf ("Instruction::ReadDictionary: Failure executing regular expression.\n");
                option_value_sp.reset();
                return option_value_sp;
            }
            
            ConstString const_key (key.c_str());
            // Check value to see if it's the start of an array or dictionary.
            
            lldb::OptionValueSP value_sp;
            assert (value.empty() == false);
            assert (key.empty() == false);            

            if (value[0] == '{')
            {
                assert (value.size() == 1);
                // value is a dictionary
                value_sp = ReadDictionary (in_file, out_stream);
                if (value_sp.get() == NULL)
                {
                    option_value_sp.reset ();
                    return option_value_sp;
                }
            }
            else if (value[0] == '[')
            {
                assert (value.size() == 1);
                // value is an array
                value_sp = ReadArray (in_file, out_stream, data_type);
                if (value_sp.get() == NULL)
                {
                    option_value_sp.reset ();
                    return option_value_sp;
                }
                // We've used the data_type to read an array; re-set the type to Invalid
                data_type = OptionValue::eTypeInvalid;
            }
            else if ((value[0] == '0') && (value[1] == 'x'))
            {
                value_sp.reset (new OptionValueUInt64 (0, 0));
                value_sp->SetValueFromCString (value.c_str());
            }
            else
            {
                int len = value.size();
                if ((value[0] == '"') && (value[len-1] == '"'))
                    value = value.substr (1, len-2);
                value_sp.reset (new OptionValueString (value.c_str(), ""));
            }

         

            if (const_key == encoding_key)
            {
                // A 'data_encoding=..." is NOT a normal key-value pair; it is meta-data indicating the
                // data type of an upcoming array (usually the next bit of data to be read in).
                if (strcmp (value.c_str(), "uint32_t") == 0)
                    data_type = OptionValue::eTypeUInt64;
            }
            else
                option_value_sp->GetAsDictionary()->SetValueForKey (const_key, value_sp, false);
        }
    }
    
    return option_value_sp;
}

bool
Instruction::TestEmulation (Stream *out_stream, const char *file_name)
{
    if (!out_stream)
        return false;

    if (!file_name)
    {
        out_stream->Printf ("Instruction::TestEmulation:  Missing file_name.");
        return false;
    }
        
    FILE *test_file = fopen (file_name, "r");
    if (!test_file)
    {
        out_stream->Printf ("Instruction::TestEmulation: Attempt to open test file failed.");
        return false;
    }

    char buffer[256];
    if (!fgets (buffer, 255, test_file))
    {
        out_stream->Printf ("Instruction::TestEmulation: Error reading first line of test file.\n");
        fclose (test_file);
        return false;
    }
    
    if (strncmp (buffer, "InstructionEmulationState={", 27) != 0)
    {
        out_stream->Printf ("Instructin::TestEmulation: Test file does not contain emulation state dictionary\n");
        fclose (test_file);
        return false;
    }

    // Read all the test information from the test file into an OptionValueDictionary.

    OptionValueSP data_dictionary_sp (ReadDictionary (test_file, out_stream));
    if (data_dictionary_sp.get() == NULL)
    {
        out_stream->Printf ("Instruction::TestEmulation:  Error reading Dictionary Object.\n");
        fclose (test_file);
        return false;
    }

    fclose (test_file);

    OptionValueDictionary *data_dictionary = data_dictionary_sp->GetAsDictionary();
    static ConstString description_key ("assembly_string");
    static ConstString triple_key ("triple");

    OptionValueSP value_sp = data_dictionary->GetValueForKey (description_key);
    
    if (value_sp.get() == NULL)
    {
        out_stream->Printf ("Instruction::TestEmulation:  Test file does not contain description string.\n");
        return false;
    }

    SetDescription (value_sp->GetStringValue());
            
            
    value_sp = data_dictionary->GetValueForKey (triple_key);
    if (value_sp.get() == NULL)
    {
        out_stream->Printf ("Instruction::TestEmulation: Test file does not contain triple.\n");
        return false;
    }
    
    ArchSpec arch;
    arch.SetTriple (llvm::Triple (value_sp->GetStringValue()));

    bool success = false;
    std::auto_ptr<EmulateInstruction> insn_emulator_ap (EmulateInstruction::FindPlugin (arch, eInstructionTypeAny, NULL));
    if (insn_emulator_ap.get())
        success = insn_emulator_ap->TestEmulation (out_stream, arch, data_dictionary);

    if (success)
        out_stream->Printf ("Emulation test succeeded.");
    else
        out_stream->Printf ("Emulation test failed.");
        
    return success;
}

bool
Instruction::Emulate (const ArchSpec &arch,
                      uint32_t evaluate_options,
                      void *baton,
                      EmulateInstruction::ReadMemoryCallback read_mem_callback,
                      EmulateInstruction::WriteMemoryCallback write_mem_callback,
                      EmulateInstruction::ReadRegisterCallback read_reg_callback,
                      EmulateInstruction::WriteRegisterCallback write_reg_callback)
{
	std::auto_ptr<EmulateInstruction> insn_emulator_ap (EmulateInstruction::FindPlugin (arch, eInstructionTypeAny, NULL));
	if (insn_emulator_ap.get())
	{
		insn_emulator_ap->SetBaton (baton);
		insn_emulator_ap->SetCallbacks (read_mem_callback, write_mem_callback, read_reg_callback, write_reg_callback);
        insn_emulator_ap->SetInstruction (GetOpcode(), GetAddress(), NULL);
        return insn_emulator_ap->EvaluateInstruction (evaluate_options);
	}

    return false;
}


uint32_t
Instruction::GetData (DataExtractor &data)
{
    return m_opcode.GetData(data);
}

InstructionList::InstructionList() :
    m_instructions()
{
}

InstructionList::~InstructionList()
{
}

size_t
InstructionList::GetSize() const
{
    return m_instructions.size();
}

uint32_t
InstructionList::GetMaxOpcocdeByteSize () const
{
    uint32_t max_inst_size = 0;
    collection::const_iterator pos, end;
    for (pos = m_instructions.begin(), end = m_instructions.end();
         pos != end;
         ++pos)
    {
        uint32_t inst_size = (*pos)->GetOpcode().GetByteSize();
        if (max_inst_size < inst_size)
            max_inst_size = inst_size;
    }
    return max_inst_size;
}



InstructionSP
InstructionList::GetInstructionAtIndex (uint32_t idx) const
{
    InstructionSP inst_sp;
    if (idx < m_instructions.size())
        inst_sp = m_instructions[idx];
    return inst_sp;
}

void
InstructionList::Dump (Stream *s,
                       bool show_address,
                       bool show_bytes,
                       const ExecutionContext* exe_ctx)
{
    const uint32_t max_opcode_byte_size = GetMaxOpcocdeByteSize();
    collection::const_iterator pos, begin, end;
    for (begin = m_instructions.begin(), end = m_instructions.end(), pos = begin;
         pos != end;
         ++pos)
    {
        if (pos != begin)
            s->EOL();
        (*pos)->Dump(s, max_opcode_byte_size, show_address, show_bytes, exe_ctx);
    }
}


void
InstructionList::Clear()
{
  m_instructions.clear();
}

void
InstructionList::Append (lldb::InstructionSP &inst_sp)
{
    if (inst_sp)
        m_instructions.push_back(inst_sp);
}

uint32_t
InstructionList::GetIndexOfNextBranchInstruction(uint32_t start) const
{
    size_t num_instructions = m_instructions.size();
    
    uint32_t next_branch = UINT32_MAX;
    for (size_t i = start; i < num_instructions; i++)
    {
        if (m_instructions[i]->DoesBranch())
        {
            next_branch = i;
            break;
        }
    }
    return next_branch;
}

uint32_t
InstructionList::GetIndexOfInstructionAtLoadAddress (lldb::addr_t load_addr, Target &target)
{
    Address address;
    address.SetLoadAddress(load_addr, &target);
    uint32_t num_instructions = m_instructions.size();
    uint32_t index = UINT32_MAX;
    for (int i = 0; i < num_instructions; i++)
    {
        if (m_instructions[i]->GetAddress() == address)
        {
            index = i;
            break;
        }
    }
    return index;
}

size_t
Disassembler::ParseInstructions
(
    const ExecutionContext *exe_ctx,
    const AddressRange &range,
    Stream *error_strm_ptr
)
{
    if (exe_ctx)
    {
        Target *target = exe_ctx->GetTargetPtr();
        const addr_t byte_size = range.GetByteSize();
        if (target == NULL || byte_size == 0 || !range.GetBaseAddress().IsValid())
            return 0;

        DataBufferHeap *heap_buffer = new DataBufferHeap (byte_size, '\0');
        DataBufferSP data_sp(heap_buffer);

        Error error;
        const bool prefer_file_cache = true;
        const size_t bytes_read = target->ReadMemory (range.GetBaseAddress(), 
                                                      prefer_file_cache, 
                                                      heap_buffer->GetBytes(), 
                                                      heap_buffer->GetByteSize(), 
                                                      error);
        
        if (bytes_read > 0)
        {
            if (bytes_read != heap_buffer->GetByteSize())
                heap_buffer->SetByteSize (bytes_read);
            DataExtractor data (data_sp, 
                                m_arch.GetByteOrder(),
                                m_arch.GetAddressByteSize());
            return DecodeInstructions (range.GetBaseAddress(), data, 0, UINT32_MAX, false);
        }
        else if (error_strm_ptr)
        {
            const char *error_cstr = error.AsCString();
            if (error_cstr)
            {
                error_strm_ptr->Printf("error: %s\n", error_cstr);
            }
        }
    }
    else if (error_strm_ptr)
    {
        error_strm_ptr->PutCString("error: invalid execution context\n");
    }
    return 0;
}

size_t
Disassembler::ParseInstructions
(
    const ExecutionContext *exe_ctx,
    const Address &start,
    uint32_t num_instructions
)
{
    m_instruction_list.Clear();

    if (exe_ctx == NULL || num_instructions == 0 || !start.IsValid())
        return 0;
        
    Target *target = exe_ctx->GetTargetPtr();
    // Calculate the max buffer size we will need in order to disassemble
    const addr_t byte_size = num_instructions * m_arch.GetMaximumOpcodeByteSize();
    
    if (target == NULL || byte_size == 0)
        return 0;

    DataBufferHeap *heap_buffer = new DataBufferHeap (byte_size, '\0');
    DataBufferSP data_sp (heap_buffer);

    Error error;
    bool prefer_file_cache = true;
    const size_t bytes_read = target->ReadMemory (start, 
                                                  prefer_file_cache, 
                                                  heap_buffer->GetBytes(), 
                                                  byte_size, 
                                                  error);

    if (bytes_read == 0)
        return 0;
    DataExtractor data (data_sp,
                        m_arch.GetByteOrder(),
                        m_arch.GetAddressByteSize());

    const bool append_instructions = true;
    DecodeInstructions (start, 
                        data, 
                        0, 
                        num_instructions, 
                        append_instructions);

    return m_instruction_list.GetSize();
}

//----------------------------------------------------------------------
// Disassembler copy constructor
//----------------------------------------------------------------------
Disassembler::Disassembler(const ArchSpec& arch) :
    m_arch (arch),
    m_instruction_list(),
    m_base_addr(LLDB_INVALID_ADDRESS)
{

}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Disassembler::~Disassembler()
{
}

InstructionList &
Disassembler::GetInstructionList ()
{
    return m_instruction_list;
}

const InstructionList &
Disassembler::GetInstructionList () const
{
    return m_instruction_list;
}

//----------------------------------------------------------------------
// Class PseudoInstruction
//----------------------------------------------------------------------
PseudoInstruction::PseudoInstruction () :
    Instruction (Address(), eAddressClassUnknown),
    m_description ()
{
}

PseudoInstruction::~PseudoInstruction ()
{
}
     
bool
PseudoInstruction::DoesBranch () const
{
    // This is NOT a valid question for a pseudo instruction.
    return false;
}
    
size_t
PseudoInstruction::Decode (const lldb_private::Disassembler &disassembler,
                           const lldb_private::DataExtractor &data,
                           uint32_t data_offset)
{
    return m_opcode.GetByteSize();
}


void
PseudoInstruction::SetOpcode (size_t opcode_size, void *opcode_data)
{
    if (!opcode_data)
        return;

    switch (opcode_size)
    {
        case 8:
        {
            uint8_t value8 = *((uint8_t *) opcode_data);
            m_opcode.SetOpcode8 (value8);
            break;
         }   
        case 16:
        {
            uint16_t value16 = *((uint16_t *) opcode_data);
            m_opcode.SetOpcode16 (value16);
            break;
         }   
        case 32:
        {
            uint32_t value32 = *((uint32_t *) opcode_data);
            m_opcode.SetOpcode32 (value32);
            break;
         }   
        case 64:
        {
            uint64_t value64 = *((uint64_t *) opcode_data);
            m_opcode.SetOpcode64 (value64);
            break;
         }   
        default:
            break;
    }
}

void
PseudoInstruction::SetDescription (const char *description)
{
    if (description && strlen (description) > 0)
        m_description = description;
}
