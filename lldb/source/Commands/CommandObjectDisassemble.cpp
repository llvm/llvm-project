//===-- CommandObjectDisassemble.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectDisassemble.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"

#define DEFAULT_DISASM_BYTE_SIZE 32
#define DEFAULT_DISASM_NUM_INS  4

using namespace lldb;
using namespace lldb_private;

CommandObjectDisassemble::CommandOptions::CommandOptions (CommandInterpreter &interpreter) :
    Options(interpreter),
    num_lines_context(0),
    num_instructions (0),
    func_name(),
    cur_function (false),
    start_addr(),
    end_addr (),
    at_pc (false),
    frame_line (false),
    plugin_name (),
    arch(),
    some_location_specified (false) 
{
    OptionParsingStarting();
}

CommandObjectDisassemble::CommandOptions::~CommandOptions ()
{
}

Error
CommandObjectDisassemble::CommandOptions::SetOptionValue (uint32_t option_idx, const char *option_arg)
{
    Error error;

    char short_option = (char) m_getopt_table[option_idx].val;

    bool success;
    
    switch (short_option)
    {
    case 'm':
        show_mixed = true;
        break;

    case 'C':
        num_lines_context = Args::StringToUInt32(option_arg, 0, 0, &success);
        if (!success)
            error.SetErrorStringWithFormat ("invalid num context lines string: \"%s\"", option_arg);
        break;

    case 'c':
        num_instructions = Args::StringToUInt32(option_arg, 0, 0, &success);
        if (!success)
            error.SetErrorStringWithFormat ("invalid num of instructions string: \"%s\"", option_arg);
        break;

    case 'b':
        show_bytes = true;
        break;

    case 's':
        start_addr = Args::StringToUInt64(option_arg, LLDB_INVALID_ADDRESS, 0);
        if (start_addr == LLDB_INVALID_ADDRESS)
            start_addr = Args::StringToUInt64(option_arg, LLDB_INVALID_ADDRESS, 16);

        if (start_addr == LLDB_INVALID_ADDRESS)
            error.SetErrorStringWithFormat ("invalid start address string '%s'", option_arg);
        some_location_specified = true;
        break;
    case 'e':
        end_addr = Args::StringToUInt64(option_arg, LLDB_INVALID_ADDRESS, 0);
        if (end_addr == LLDB_INVALID_ADDRESS)
            end_addr = Args::StringToUInt64(option_arg, LLDB_INVALID_ADDRESS, 16);

        if (end_addr == LLDB_INVALID_ADDRESS)
            error.SetErrorStringWithFormat ("invalid end address string '%s'", option_arg);
        break;
        some_location_specified = true;
    case 'n':
        func_name.assign (option_arg);
        some_location_specified = true;
        break;

    case 'p':
        at_pc = true;
        some_location_specified = true;
        break;

    case 'l':
        frame_line = true;
        // Disassemble the current source line kind of implies showing mixed
        // source code context. 
        show_mixed = true;
        some_location_specified = true;
        break;

    case 'P':
        plugin_name.assign (option_arg);
        break;

    case 'r':
        raw = true;
        break;

    case 'f':
        cur_function = true;
        some_location_specified = true;
        break;

    case 'a':
        if (!arch.SetTriple (option_arg, m_interpreter.GetPlatform (true).get()))
            arch.SetTriple (option_arg);
        break;

    default:
        error.SetErrorStringWithFormat("unrecognized short option '%c'", short_option);
        break;
    }

    return error;
}

void
CommandObjectDisassemble::CommandOptions::OptionParsingStarting ()
{
    show_mixed = false;
    show_bytes = false;
    num_lines_context = 0;
    num_instructions = 0;
    func_name.clear();
    cur_function = false;
    at_pc = false;
    frame_line = false;
    start_addr = LLDB_INVALID_ADDRESS;
    end_addr = LLDB_INVALID_ADDRESS;
    raw = false;
    plugin_name.clear();
    arch.Clear();
    some_location_specified = false;
}

Error
CommandObjectDisassemble::CommandOptions::OptionParsingFinished ()
{
    if (!some_location_specified)
        at_pc = true;
    return Error();
    
}

const OptionDefinition*
CommandObjectDisassemble::CommandOptions::GetDefinitions ()
{
    return g_option_table;
}

OptionDefinition
CommandObjectDisassemble::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_ALL  , false , "bytes",          'b', no_argument        , NULL, 0, eArgTypeNone,        "Show opcode bytes when disassembling."},
{ LLDB_OPT_SET_ALL  , false , "context",        'C', required_argument  , NULL, 0, eArgTypeNumLines,    "Number of context lines of source to show."},
{ LLDB_OPT_SET_ALL  , false , "mixed",          'm', no_argument        , NULL, 0, eArgTypeNone,        "Enable mixed source and assembly display."},
{ LLDB_OPT_SET_ALL  , false , "raw",            'r', no_argument        , NULL, 0, eArgTypeNone,        "Print raw disassembly with no symbol information."},
{ LLDB_OPT_SET_ALL  , false , "plugin",         'P', required_argument  , NULL, 0, eArgTypePlugin,      "Name of the disassembler plugin you want to use."},
{ LLDB_OPT_SET_ALL  , false , "arch",           'a', required_argument  , NULL, 0, eArgTypeArchitecture,"Specify the architecture to use from cross disassembly."},
{ LLDB_OPT_SET_1 |
  LLDB_OPT_SET_2    , true  , "start-address" , 's', required_argument  , NULL, 0, eArgTypeStartAddress,"Address at which to start disassembling."},
{ LLDB_OPT_SET_1    , false , "end-address"  ,  'e', required_argument  , NULL, 0, eArgTypeEndAddress,  "Address at which to end disassembling."},
{ LLDB_OPT_SET_2 |
  LLDB_OPT_SET_3 |
  LLDB_OPT_SET_4 |
  LLDB_OPT_SET_5    , false , "count",          'c', required_argument  , NULL, 0, eArgTypeNumLines,    "Number of instructions to display."},
{ LLDB_OPT_SET_3    , false  , "name",           'n', required_argument  , NULL, CommandCompletions::eSymbolCompletion, eArgTypeFunctionName,             "Disassemble entire contents of the given function name."},
{ LLDB_OPT_SET_4    , false  , "frame",          'f', no_argument        , NULL, 0, eArgTypeNone,        "Disassemble from the start of the current frame's function."},
{ LLDB_OPT_SET_5    , false  , "pc",             'p', no_argument        , NULL, 0, eArgTypeNone,        "Disassemble around the current pc."},
{ LLDB_OPT_SET_6    , false  , "line",           'l', no_argument        , NULL, 0, eArgTypeNone,        "Disassemble the current frame's current source line instructions if there debug line table information, else disasemble around the pc."},
{ 0                 , false , NULL,             0,   0                  , NULL, 0, eArgTypeNone,        NULL }
};



//-------------------------------------------------------------------------
// CommandObjectDisassemble
//-------------------------------------------------------------------------

CommandObjectDisassemble::CommandObjectDisassemble (CommandInterpreter &interpreter) :
    CommandObjectParsed (interpreter,
                         "disassemble",
                         "Disassemble bytes in the current function, or elsewhere in the executable program as specified by the user.",
                         "disassemble [<cmd-options>]"),
    m_options (interpreter)
{
}

CommandObjectDisassemble::~CommandObjectDisassemble()
{
}

bool
CommandObjectDisassemble::DoExecute (Args& command, CommandReturnObject &result)
{
    Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
    if (target == NULL)
    {
        result.AppendError ("invalid target, create a debug target using the 'target create' command");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }
    if (!m_options.arch.IsValid())
        m_options.arch = target->GetArchitecture();

    if (!m_options.arch.IsValid())
    {
        result.AppendError ("use the --arch option or set the target architecure to disassemble");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    const char *plugin_name = m_options.GetPluginName ();
    DisassemblerSP disassembler = Disassembler::FindPlugin(m_options.arch, plugin_name);

    if (!disassembler)
    {
        if (plugin_name)
            result.AppendErrorWithFormat ("Unable to find Disassembler plug-in named '%s' that supports the '%s' architecture.\n", 
                                          plugin_name,
                                          m_options.arch.GetArchitectureName());
        else
            result.AppendErrorWithFormat ("Unable to find Disassembler plug-in for the '%s' architecture.\n", 
                                          m_options.arch.GetArchitectureName());
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    result.SetStatus (eReturnStatusSuccessFinishResult);

    if (command.GetArgumentCount() != 0)
    {
        result.AppendErrorWithFormat ("\"disassemble\" arguments are specified as options.\n");
        GetOptions()->GenerateOptionUsage (result.GetErrorStream(), this);
        result.SetStatus (eReturnStatusFailed);
        return false;
    }
    
    if (m_options.show_mixed && m_options.num_lines_context == 0)
        m_options.num_lines_context = 1;

    ExecutionContext exe_ctx(m_interpreter.GetExecutionContext());
    // Always show the PC in the disassembly
    uint32_t options = Disassembler::eOptionMarkPCAddress;

    // Mark the source line for the current PC only if we are doing mixed source and assembly
    if (m_options.show_mixed)
        options |= Disassembler::eOptionMarkPCSourceLine;

    if (m_options.show_bytes)
        options |= Disassembler::eOptionShowBytes;

    if (m_options.raw)
        options |= Disassembler::eOptionRawOuput;

    if (!m_options.func_name.empty())
    {
        ConstString name(m_options.func_name.c_str());
        
        if (Disassembler::Disassemble (m_interpreter.GetDebugger(), 
                                       m_options.arch,
                                       plugin_name,
                                       exe_ctx,
                                       name,
                                       NULL,    // Module *
                                       m_options.num_instructions,
                                       m_options.show_mixed ? m_options.num_lines_context : 0,
                                       options,
                                       result.GetOutputStream()))
        {
            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        else
        {
            result.AppendErrorWithFormat ("Unable to find symbol with name '%s'.\n", name.GetCString());
            result.SetStatus (eReturnStatusFailed);
        }
    } 
    else
    {
        AddressRange range;
        StackFrame *frame = exe_ctx.GetFramePtr();
        if (m_options.frame_line)
        {
            if (frame == NULL)
            {
                result.AppendError ("Cannot disassemble around the current line without a selected frame.\n");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            LineEntry pc_line_entry (frame->GetSymbolContext(eSymbolContextLineEntry).line_entry);
            if (pc_line_entry.IsValid())
            {
                range = pc_line_entry.range;
            }
            else
            {
                m_options.at_pc = true; // No line entry, so just disassemble around the current pc
                m_options.show_mixed = false;
            }
        }
        else if (m_options.cur_function)
        {
            if (frame == NULL)
            {
                result.AppendError ("Cannot disassemble around the current function without a selected frame.\n");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            Symbol *symbol = frame->GetSymbolContext(eSymbolContextSymbol).symbol;
            if (symbol)
            {
                range.GetBaseAddress() = symbol->GetAddress();
                range.SetByteSize(symbol->GetByteSize());
            }
        }

        // Did the "m_options.frame_line" find a valid range already? If so
        // skip the rest...
        if (range.GetByteSize() == 0)
        {
            if (m_options.at_pc)
            {
                if (frame == NULL)
                {
                    result.AppendError ("Cannot disassemble around the current PC without a selected frame.\n");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                range.GetBaseAddress() = frame->GetFrameCodeAddress();
                if (m_options.num_instructions == 0)
                {
                    // Disassembling at the PC always disassembles some number of instructions (not the whole function).
                    m_options.num_instructions = DEFAULT_DISASM_NUM_INS;
                }
            }
            else
            {
                range.GetBaseAddress().SetOffset (m_options.start_addr);
                if (range.GetBaseAddress().IsValid())
                {
                    if (m_options.end_addr != LLDB_INVALID_ADDRESS)
                    {
                        if (m_options.end_addr <= m_options.start_addr)
                        {
                            result.AppendErrorWithFormat ("End address before start address.\n");
                            result.SetStatus (eReturnStatusFailed);
                            return false;            
                        }
                        range.SetByteSize (m_options.end_addr - m_options.start_addr);
                    }
                }
            }
        }
        
        if (m_options.num_instructions != 0)
        {
            if (!range.GetBaseAddress().IsValid())
            {
                // The default action is to disassemble the current frame function.
                if (frame)
                {
                    SymbolContext sc(frame->GetSymbolContext(eSymbolContextFunction | eSymbolContextSymbol));
                    if (sc.function)
                        range.GetBaseAddress() = sc.function->GetAddressRange().GetBaseAddress();
                    else if (sc.symbol && sc.symbol->ValueIsAddress())
                        range.GetBaseAddress() = sc.symbol->GetAddress();
                    else
                        range.GetBaseAddress() = frame->GetFrameCodeAddress();
                }
                
                if (!range.GetBaseAddress().IsValid())
                {
                    result.AppendError ("invalid frame");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }

            if (Disassembler::Disassemble (m_interpreter.GetDebugger(), 
                                           m_options.arch,
                                           plugin_name,
                                           exe_ctx,
                                           range.GetBaseAddress(),
                                           m_options.num_instructions,
                                           m_options.show_mixed ? m_options.num_lines_context : 0,
                                           options,
                                           result.GetOutputStream()))
            {
                result.SetStatus (eReturnStatusSuccessFinishResult);
            }
            else
            {
                result.AppendErrorWithFormat ("Failed to disassemble memory at 0x%8.8llx.\n", m_options.start_addr);
                result.SetStatus (eReturnStatusFailed);            
            }
        }
        else
        {
            if (!range.GetBaseAddress().IsValid())
            {
                // The default action is to disassemble the current frame function.
                if (frame)
                {
                    SymbolContext sc(frame->GetSymbolContext(eSymbolContextFunction | eSymbolContextSymbol));
                    if (sc.function)
                        range = sc.function->GetAddressRange();
                    else if (sc.symbol && sc.symbol->ValueIsAddress())
                    {
                        range.GetBaseAddress() = sc.symbol->GetAddress();
                        range.SetByteSize (sc.symbol->GetByteSize());
                    }
                    else
                        range.GetBaseAddress() = frame->GetFrameCodeAddress();
                }
                else
                {
                    result.AppendError ("invalid frame");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            if (range.GetByteSize() == 0)
                range.SetByteSize(DEFAULT_DISASM_BYTE_SIZE);

            if (Disassembler::Disassemble (m_interpreter.GetDebugger(), 
                                           m_options.arch,
                                           plugin_name,
                                           exe_ctx,
                                           range,
                                           m_options.num_instructions,
                                           m_options.show_mixed ? m_options.num_lines_context : 0,
                                           options,
                                           result.GetOutputStream()))
            {
                result.SetStatus (eReturnStatusSuccessFinishResult);
            }
            else
            {
                result.AppendErrorWithFormat ("Failed to disassemble memory at 0x%8.8llx.\n", m_options.start_addr);
                result.SetStatus (eReturnStatusFailed);            
            }
        }
    }

    return result.Succeeded();
}
