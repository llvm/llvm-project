//===-- CommandObjectMemory.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "CommandObjectMemory.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObjectMemory.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/OptionGroupFormat.h"
#include "lldb/Interpreter/OptionGroupOutputFile.h"
#include "lldb/Interpreter/OptionGroupValueObjectDisplay.h"
#include "lldb/Interpreter/OptionValueString.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"

using namespace lldb;
using namespace lldb_private;

static OptionDefinition
g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "num-per-line" ,'l', required_argument, NULL, 0, eArgTypeNumberPerLine ,"The number of items per line to display."},
    { LLDB_OPT_SET_2, false, "binary"       ,'b', no_argument      , NULL, 0, eArgTypeNone          ,"If true, memory will be saved as binary. If false, the memory is saved save as an ASCII dump that uses the format, size, count and number per line settings."},
    { LLDB_OPT_SET_3, true , "type"         ,'t', required_argument, NULL, 0, eArgTypeNone          ,"The name of a type to view memory as."}, 
    { LLDB_OPT_SET_1|
      LLDB_OPT_SET_2|
      LLDB_OPT_SET_3, false, "force"        ,'r', no_argument,       NULL, 0, eArgTypeNone          ,"Necessary if reading over 1024 bytes of memory."},
};



class OptionGroupReadMemory : public OptionGroup
{
public:

    OptionGroupReadMemory () :
        m_num_per_line (1,1),
        m_output_as_binary (false),
        m_view_as_type()
    {
    }

    virtual
    ~OptionGroupReadMemory ()
    {
    }
    
    
    virtual uint32_t
    GetNumDefinitions ()
    {
        return sizeof (g_option_table) / sizeof (OptionDefinition);
    }
    
    virtual const OptionDefinition*
    GetDefinitions ()
    {
        return g_option_table;
    }
    
    virtual Error
    SetOptionValue (CommandInterpreter &interpreter,
                    uint32_t option_idx,
                    const char *option_arg)
    {
        Error error;
        const int short_option = g_option_table[option_idx].short_option;
        
        switch (short_option)
        {
            case 'l':
                error = m_num_per_line.SetValueFromCString (option_arg);
                if (m_num_per_line.GetCurrentValue() == 0)
                    error.SetErrorStringWithFormat("invalid value for --num-per-line option '%s'", option_arg);
                break;

            case 'b':
                m_output_as_binary = true;
                break;
                
            case 't':
                error = m_view_as_type.SetValueFromCString (option_arg);
                break;
            
            case 'r':
                m_force = true;
                break;
                
            default:
                error.SetErrorStringWithFormat("unrecognized short option '%c'", short_option);
                break;
        }
        return error;
    }
    
    virtual void
    OptionParsingStarting (CommandInterpreter &interpreter)
    {
        m_num_per_line.Clear();
        m_output_as_binary = false;
        m_view_as_type.Clear();
    }
    
    Error
    FinalizeSettings (Target *target, OptionGroupFormat& format_options)
    {
        Error error;
        OptionValueUInt64 &byte_size_value = format_options.GetByteSizeValue();
        OptionValueUInt64 &count_value = format_options.GetCountValue();
        const bool byte_size_option_set = byte_size_value.OptionWasSet();
        const bool num_per_line_option_set = m_num_per_line.OptionWasSet();
        const bool count_option_set = format_options.GetCountValue().OptionWasSet();
        
        switch (format_options.GetFormat())
        {
            default:
                break;
                
            case eFormatBoolean:
                if (!byte_size_option_set)
                    byte_size_value = 1;
                if (!num_per_line_option_set)
                    m_num_per_line = 1;
                if (!count_option_set)
                    format_options.GetCountValue() = 8;
                break;
                
            case eFormatCString:
                break;

            case eFormatInstruction:
                if (count_option_set)
                    byte_size_value = target->GetArchitecture().GetMaximumOpcodeByteSize();
                m_num_per_line = 1;
                break;

            case eFormatAddressInfo:
                if (!byte_size_option_set)
                    byte_size_value = target->GetArchitecture().GetAddressByteSize();
                m_num_per_line = 1;
                if (!count_option_set)
                    format_options.GetCountValue() = 8;
                break;

            case eFormatPointer:
                byte_size_value = target->GetArchitecture().GetAddressByteSize();
                if (!num_per_line_option_set)
                    m_num_per_line = 4;
                if (!count_option_set)
                    format_options.GetCountValue() = 8;
                break;
                
            case eFormatBinary:
            case eFormatFloat:
            case eFormatOctal:
            case eFormatDecimal:
            case eFormatEnum:
            case eFormatUnicode16:
            case eFormatUnicode32:
            case eFormatUnsigned:
            case eFormatHexFloat:
                if (!byte_size_option_set)
                    byte_size_value = 4;
                if (!num_per_line_option_set)
                    m_num_per_line = 1;
                if (!count_option_set)
                    format_options.GetCountValue() = 8;
                break;
            
            case eFormatBytes:
            case eFormatBytesWithASCII:
                if (byte_size_option_set)
                {
                    if (byte_size_value > 1)
                        error.SetErrorStringWithFormat ("display format (bytes/bytes with ascii) conflicts with the specified byte size %" PRIu64 "\n"
                                                        "\tconsider using a different display format or don't specify the byte size",
                                                        byte_size_value.GetCurrentValue());
                }
                else
                    byte_size_value = 1;
                if (!num_per_line_option_set)
                    m_num_per_line = 16;
                if (!count_option_set)
                    format_options.GetCountValue() = 32;
                break;
            case eFormatCharArray:
            case eFormatChar:
            case eFormatCharPrintable:
                if (!byte_size_option_set)
                    byte_size_value = 1;
                if (!num_per_line_option_set)
                    m_num_per_line = 32;
                if (!count_option_set)
                    format_options.GetCountValue() = 64;
                break;
            case eFormatComplex:
                if (!byte_size_option_set)
                    byte_size_value = 8;
                if (!num_per_line_option_set)
                    m_num_per_line = 1;
                if (!count_option_set)
                    format_options.GetCountValue() = 8;
                break;
            case eFormatComplexInteger:
                if (!byte_size_option_set)
                    byte_size_value = 8;
                if (!num_per_line_option_set)
                    m_num_per_line = 1;
                if (!count_option_set)
                    format_options.GetCountValue() = 8;
                break;
            case eFormatHex:
                if (!byte_size_option_set)
                    byte_size_value = 4;
                if (!num_per_line_option_set)
                {
                    switch (byte_size_value)
                    {
                        case 1:
                        case 2:
                            m_num_per_line = 8;
                            break;
                        case 4:
                            m_num_per_line = 4;
                            break;
                        case 8:
                            m_num_per_line = 2;
                            break;
                        default:
                            m_num_per_line = 1;
                            break;
                    }
                }
                if (!count_option_set)
                    count_value = 8;
                break;
                
            case eFormatVectorOfChar:
            case eFormatVectorOfSInt8:
            case eFormatVectorOfUInt8:
            case eFormatVectorOfSInt16:
            case eFormatVectorOfUInt16:
            case eFormatVectorOfSInt32:
            case eFormatVectorOfUInt32:
            case eFormatVectorOfSInt64:
            case eFormatVectorOfUInt64:
            case eFormatVectorOfFloat32:
            case eFormatVectorOfFloat64:
            case eFormatVectorOfUInt128:
                if (!byte_size_option_set)
                    byte_size_value = 128;
                if (!num_per_line_option_set)
                    m_num_per_line = 1;
                if (!count_option_set)
                    count_value = 4;
                break;
        }
        return error;
    }

    bool
    AnyOptionWasSet () const
    {
        return m_num_per_line.OptionWasSet() ||
               m_output_as_binary ||
               m_view_as_type.OptionWasSet();
    }
    
    OptionValueUInt64 m_num_per_line;
    bool m_output_as_binary;
    OptionValueString m_view_as_type;
    bool m_force;
};



//----------------------------------------------------------------------
// Read memory from the inferior process
//----------------------------------------------------------------------
class CommandObjectMemoryRead : public CommandObjectParsed
{
public:

    CommandObjectMemoryRead (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "memory read",
                             "Read from the memory of the process being debugged.",
                             NULL,
                             eFlagRequiresTarget | eFlagProcessMustBePaused),
        m_option_group (interpreter),
        m_format_options (eFormatBytesWithASCII, 1, 8),
        m_memory_options (),
        m_outfile_options (),
        m_varobj_options(),
        m_next_addr(LLDB_INVALID_ADDRESS),
        m_prev_byte_size(0),
        m_prev_format_options (eFormatBytesWithASCII, 1, 8),
        m_prev_memory_options (),
        m_prev_outfile_options (),
        m_prev_varobj_options()
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData start_addr_arg;
        CommandArgumentData end_addr_arg;
        
        // Define the first (and only) variant of this arg.
        start_addr_arg.arg_type = eArgTypeAddressOrExpression;
        start_addr_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (start_addr_arg);
        
        // Define the first (and only) variant of this arg.
        end_addr_arg.arg_type = eArgTypeAddressOrExpression;
        end_addr_arg.arg_repetition = eArgRepeatOptional;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg2.push_back (end_addr_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
        
        // Add the "--format" and "--count" options to group 1 and 3
        m_option_group.Append (&m_format_options, 
                               OptionGroupFormat::OPTION_GROUP_FORMAT | OptionGroupFormat::OPTION_GROUP_COUNT, 
                               LLDB_OPT_SET_1 | LLDB_OPT_SET_2 | LLDB_OPT_SET_3);
        m_option_group.Append (&m_format_options, 
                               OptionGroupFormat::OPTION_GROUP_GDB_FMT, 
                               LLDB_OPT_SET_1 | LLDB_OPT_SET_3);
        // Add the "--size" option to group 1 and 2
        m_option_group.Append (&m_format_options, 
                               OptionGroupFormat::OPTION_GROUP_SIZE, 
                               LLDB_OPT_SET_1 | LLDB_OPT_SET_2);
        m_option_group.Append (&m_memory_options);
        m_option_group.Append (&m_outfile_options, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1 | LLDB_OPT_SET_2 | LLDB_OPT_SET_3);
        m_option_group.Append (&m_varobj_options, LLDB_OPT_SET_ALL, LLDB_OPT_SET_3);
        m_option_group.Finalize();
    }

    virtual
    ~CommandObjectMemoryRead ()
    {
    }

    Options *
    GetOptions ()
    {
        return &m_option_group;
    }

    virtual const char *GetRepeatCommand (Args &current_command_args, uint32_t index)
    {
        return m_cmd_name.c_str();
    }

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        // No need to check "target" for validity as eFlagRequiresTarget ensures it is valid
        Target *target = m_exe_ctx.GetTargetPtr();

        const size_t argc = command.GetArgumentCount();

        if ((argc == 0 && m_next_addr == LLDB_INVALID_ADDRESS) || argc > 2)
        {
            result.AppendErrorWithFormat ("%s takes a start address expression with an optional end address expression.\n", m_cmd_name.c_str());
            result.AppendRawWarning("Expressions should be quoted if they contain spaces or other special characters.\n");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        ClangASTType clang_ast_type;        
        Error error;

        const char *view_as_type_cstr = m_memory_options.m_view_as_type.GetCurrentValue();
        if (view_as_type_cstr && view_as_type_cstr[0])
        {
            // We are viewing memory as a type
            
            SymbolContext sc;
            const bool exact_match = false;
            TypeList type_list;
            uint32_t reference_count = 0;
            uint32_t pointer_count = 0;
            size_t idx;
            
#define ALL_KEYWORDS        \
    KEYWORD("const")        \
    KEYWORD("volatile")     \
    KEYWORD("restrict")     \
    KEYWORD("struct")       \
    KEYWORD("class")        \
    KEYWORD("union")
            
#define KEYWORD(s) s,
            static const char *g_keywords[] =
            {
                ALL_KEYWORDS
            };
#undef KEYWORD

#define KEYWORD(s) (sizeof(s) - 1),
            static const int g_keyword_lengths[] =
            {
                ALL_KEYWORDS
            };
#undef KEYWORD
            
#undef ALL_KEYWORDS
            
            static size_t g_num_keywords = sizeof(g_keywords) / sizeof(const char *);
            std::string type_str(view_as_type_cstr);
            
            // Remove all instances of g_keywords that are followed by spaces
            for (size_t i = 0; i < g_num_keywords; ++i)
            {
                const char *keyword = g_keywords[i];
                int keyword_len = g_keyword_lengths[i];
                
                idx = 0;
                while ((idx = type_str.find (keyword, idx)) != std::string::npos)
                {
                    if (type_str[idx + keyword_len] == ' ' || type_str[idx + keyword_len] == '\t')
                    {
                        type_str.erase(idx, keyword_len+1);
                        idx = 0;
                    }
                    else
                    {
                        idx += keyword_len;
                    }
                }
            }
            bool done = type_str.empty();
            // 
            idx = type_str.find_first_not_of (" \t");
            if (idx > 0 && idx != std::string::npos)
                type_str.erase (0, idx);
            while (!done)
            {
                // Strip trailing spaces
                if (type_str.empty())
                    done = true;
                else
                {
                    switch (type_str[type_str.size()-1])
                    {
                    case '*':
                        ++pointer_count;
                        // fall through...
                    case ' ':
                    case '\t':
                        type_str.erase(type_str.size()-1);
                        break;

                    case '&':
                        if (reference_count == 0)
                        {
                            reference_count = 1;
                            type_str.erase(type_str.size()-1);
                        }
                        else
                        {
                            result.AppendErrorWithFormat ("invalid type string: '%s'\n", view_as_type_cstr);
                            result.SetStatus(eReturnStatusFailed);
                            return false;
                        }
                        break;

                    default:
                        done = true;
                        break;
                    }
                }
            }
                    
            ConstString lookup_type_name(type_str.c_str());
            StackFrame *frame = m_exe_ctx.GetFramePtr();
            if (frame)
            {
                sc = frame->GetSymbolContext (eSymbolContextModule);
                if (sc.module_sp)
                {
                    sc.module_sp->FindTypes (sc,
                                             lookup_type_name,
                                             exact_match,
                                             1, 
                                             type_list);
                }
            }
            if (type_list.GetSize() == 0)
            {
                target->GetImages().FindTypes (sc, 
                                               lookup_type_name, 
                                               exact_match, 
                                               1, 
                                               type_list);
            }
            
            if (type_list.GetSize() == 0)
            {
                result.AppendErrorWithFormat ("unable to find any types that match the raw type '%s' for full type '%s'\n", 
                                              lookup_type_name.GetCString(), 
                                              view_as_type_cstr);
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
            
            TypeSP type_sp (type_list.GetTypeAtIndex(0));
            clang_ast_type.SetClangType (type_sp->GetClangAST(), type_sp->GetClangFullType());
            
            while (pointer_count > 0)
            {
                clang_type_t pointer_type = ClangASTContext::CreatePointerType (clang_ast_type.GetASTContext(), clang_ast_type.GetOpaqueQualType());
                if (pointer_type)
                    clang_ast_type.SetClangType (clang_ast_type.GetASTContext(), pointer_type);
                else
                {
                    result.AppendError ("unable make a pointer type\n");
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                --pointer_count;
            }

            m_format_options.GetByteSizeValue() = clang_ast_type.GetClangTypeByteSize();
            
            if (m_format_options.GetByteSizeValue() == 0)
            {
                result.AppendErrorWithFormat ("unable to get the byte size of the type '%s'\n", 
                                              view_as_type_cstr);
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
            
            if (!m_format_options.GetCountValue().OptionWasSet())
                m_format_options.GetCountValue() = 1;
        }
        else
        {
            error = m_memory_options.FinalizeSettings (target, m_format_options);
        }

        // Look for invalid combinations of settings
        if (error.Fail())
        {
            result.AppendError(error.AsCString());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        lldb::addr_t addr;
        size_t total_byte_size = 0;
        if (argc == 0)
        {
            // Use the last address and byte size and all options as they were
            // if no options have been set
            addr = m_next_addr;
            total_byte_size = m_prev_byte_size;
            clang_ast_type = m_prev_clang_ast_type;
            if (!m_format_options.AnyOptionWasSet() &&
                !m_memory_options.AnyOptionWasSet() &&
                !m_outfile_options.AnyOptionWasSet() &&
                !m_varobj_options.AnyOptionWasSet())
            {
                m_format_options = m_prev_format_options;
                m_memory_options = m_prev_memory_options;
                m_outfile_options = m_prev_outfile_options;
                m_varobj_options = m_prev_varobj_options;
            }
        }

        size_t item_count = m_format_options.GetCountValue().GetCurrentValue();
        size_t item_byte_size = m_format_options.GetByteSizeValue().GetCurrentValue();
        const size_t num_per_line = m_memory_options.m_num_per_line.GetCurrentValue();

        if (total_byte_size == 0)
        {
            total_byte_size = item_count * item_byte_size;
            if (total_byte_size == 0)
                total_byte_size = 32;
        }

        if (argc > 0)
            addr = Args::StringToAddress(&m_exe_ctx, command.GetArgumentAtIndex(0), LLDB_INVALID_ADDRESS, &error);

        if (addr == LLDB_INVALID_ADDRESS)
        {
            result.AppendError("invalid start address expression.");
            result.AppendError(error.AsCString());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        if (argc == 2)
        {
            lldb::addr_t end_addr = Args::StringToAddress(&m_exe_ctx, command.GetArgumentAtIndex(1), LLDB_INVALID_ADDRESS, 0);
            if (end_addr == LLDB_INVALID_ADDRESS)
            {
                result.AppendError("invalid end address expression.");
                result.AppendError(error.AsCString());
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
            else if (end_addr <= addr)
            {
                result.AppendErrorWithFormat("end address (0x%" PRIx64 ") must be greater that the start address (0x%" PRIx64 ").\n", end_addr, addr);
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
            else if (m_format_options.GetCountValue().OptionWasSet())
            {
                result.AppendErrorWithFormat("specify either the end address (0x%" PRIx64 ") or the count (--count %lu), not both.\n", end_addr, item_count);
                result.SetStatus(eReturnStatusFailed);
                return false;
            }

            total_byte_size = end_addr - addr;
            item_count = total_byte_size / item_byte_size;
        }
        
        if (total_byte_size > 1024 && !m_memory_options.m_force)
        {
            result.AppendErrorWithFormat("Normally, \'memory read\' will not read over 1Kbyte of data.\n");
            result.AppendErrorWithFormat("Please use --force to override this restriction.\n");
            return false;
        }
        
        
        
        DataBufferSP data_sp;
        size_t bytes_read = 0;
        if (clang_ast_type.GetOpaqueQualType())
        {
            // Make sure we don't display our type as ASCII bytes like the default memory read
            if (m_format_options.GetFormatValue().OptionWasSet() == false)
                m_format_options.GetFormatValue().SetCurrentValue(eFormatDefault);

            bytes_read = clang_ast_type.GetTypeByteSize() * m_format_options.GetCountValue().GetCurrentValue();
        }
        else if (m_format_options.GetFormatValue().GetCurrentValue() != eFormatCString)
        {
            data_sp.reset (new DataBufferHeap (total_byte_size, '\0'));
            Address address(addr, NULL);
            bytes_read = target->ReadMemory(address, false, data_sp->GetBytes (), data_sp->GetByteSize(), error);
            if (bytes_read == 0)
            {
                const char *error_cstr = error.AsCString();
                if (error_cstr && error_cstr[0])
                {
                    result.AppendError(error_cstr);
                }
                else
                {
                    result.AppendErrorWithFormat("failed to read memory from 0x%" PRIx64 ".\n", addr);
                }
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
            
            if (bytes_read < total_byte_size)
                result.AppendWarningWithFormat("Not all bytes (%lu/%lu) were able to be read from 0x%" PRIx64 ".\n", bytes_read, total_byte_size, addr);
        }
        else
        {
            // we treat c-strings as a special case because they do not have a fixed size
            if (m_format_options.GetByteSizeValue().OptionWasSet() && !m_format_options.HasGDBFormat())
                item_byte_size = m_format_options.GetByteSizeValue().GetCurrentValue();
            else
                item_byte_size = target->GetMaximumSizeOfStringSummary();
            if (!m_format_options.GetCountValue().OptionWasSet())
                item_count = 1;
            data_sp.reset (new DataBufferHeap ((item_byte_size+1) * item_count, '\0')); // account for NULLs as necessary
            uint8_t *data_ptr = data_sp->GetBytes();
            auto data_addr = addr;
            auto count = item_count;
            item_count = 0;
            while (item_count < count)
            {
                std::string buffer;
                buffer.resize(item_byte_size+1,0);
                Error error;
                size_t read = target->ReadCStringFromMemory(data_addr, &buffer[0], item_byte_size+1, error);
                if (error.Fail())
                {
                    result.AppendErrorWithFormat("failed to read memory from 0x%" PRIx64 ".\n", addr);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                if (item_byte_size == read)
                {
                    result.AppendWarningWithFormat("unable to find a NULL terminated string at 0x%" PRIx64 ".Consider increasing the maximum read length.\n", data_addr);
                    break;
                }
                read+=1; // account for final NULL byte
                memcpy(data_ptr, &buffer[0], read);
                data_ptr += read;
                data_addr += read;
                bytes_read += read;
                item_count++; // if we break early we know we only read item_count strings
            }
            data_sp.reset(new DataBufferHeap(data_sp->GetBytes(),bytes_read+1));
        }

        m_next_addr = addr + bytes_read;
        m_prev_byte_size = bytes_read;
        m_prev_format_options = m_format_options;
        m_prev_memory_options = m_memory_options;
        m_prev_outfile_options = m_outfile_options;
        m_prev_varobj_options = m_varobj_options;
        m_prev_clang_ast_type = clang_ast_type;

        StreamFile outfile_stream;
        Stream *output_stream = NULL;
        const FileSpec &outfile_spec = m_outfile_options.GetFile().GetCurrentValue();
        if (outfile_spec)
        {
            char path[PATH_MAX];
            outfile_spec.GetPath (path, sizeof(path));
            
            uint32_t open_options = File::eOpenOptionWrite | File::eOpenOptionCanCreate;
            const bool append = m_outfile_options.GetAppend().GetCurrentValue();
            if (append)
                open_options |= File::eOpenOptionAppend;
            
            if (outfile_stream.GetFile ().Open (path, open_options).Success())
            {
                if (m_memory_options.m_output_as_binary)
                {
                    const size_t bytes_written = outfile_stream.Write (data_sp->GetBytes(), bytes_read);
                    if (bytes_written > 0)
                    {
                        result.GetOutputStream().Printf ("%zi bytes %s to '%s'\n", 
                                                         bytes_written, 
                                                         append ? "appended" : "written", 
                                                         path);
                        return true;
                    }
                    else 
                    {
                        result.AppendErrorWithFormat("Failed to write %" PRIu64 " bytes to '%s'.\n", (uint64_t)bytes_read, path);
                        result.SetStatus(eReturnStatusFailed);
                        return false;
                    }
                }
                else
                {
                    // We are going to write ASCII to the file just point the
                    // output_stream to our outfile_stream...
                    output_stream = &outfile_stream;
                }
            }
            else 
            {
                result.AppendErrorWithFormat("Failed to open file '%s' for %s.\n", path, append ? "append" : "write");
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
        }
        else 
        {
            output_stream = &result.GetOutputStream();
        }


        ExecutionContextScope *exe_scope = m_exe_ctx.GetBestExecutionContextScope();
        if (clang_ast_type.GetOpaqueQualType())
        {
            for (uint32_t i = 0; i<item_count; ++i)
            {
                addr_t item_addr = addr + (i * item_byte_size);
                Address address (item_addr);
                StreamString name_strm;
                name_strm.Printf ("0x%" PRIx64, item_addr);
                ValueObjectSP valobj_sp (ValueObjectMemory::Create (exe_scope, 
                                                                    name_strm.GetString().c_str(), 
                                                                    address, 
                                                                    clang_ast_type));
                if (valobj_sp)
                {
                    Format format = m_format_options.GetFormat();
                    if (format != eFormatDefault)
                        valobj_sp->SetFormat (format);

                    ValueObject::DumpValueObjectOptions options(m_varobj_options.GetAsDumpOptions(false,format));
                    
                    ValueObject::DumpValueObject (*output_stream,
                                                  valobj_sp.get(),
                                                  options);
                }
                else
                {
                    result.AppendErrorWithFormat ("failed to create a value object for: (%s) %s\n", 
                                                  view_as_type_cstr, 
                                                  name_strm.GetString().c_str());
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
            }
            return true;
        }

        result.SetStatus(eReturnStatusSuccessFinishResult);
        DataExtractor data (data_sp, 
                            target->GetArchitecture().GetByteOrder(), 
                            target->GetArchitecture().GetAddressByteSize());


        assert (output_stream);
        size_t bytes_dumped = data.Dump (output_stream,
                                         0,
                                         m_format_options.GetFormat(),
                                         item_byte_size,
                                         item_count,
                                         num_per_line,
                                         addr,
                                         0,
                                         0,
                                         exe_scope);
        m_next_addr = addr + bytes_dumped;
        output_stream->EOL();
        return true;
    }

    OptionGroupOptions m_option_group;
    OptionGroupFormat m_format_options;
    OptionGroupReadMemory m_memory_options;
    OptionGroupOutputFile m_outfile_options;
    OptionGroupValueObjectDisplay m_varobj_options;
    lldb::addr_t m_next_addr;
    lldb::addr_t m_prev_byte_size; 
    OptionGroupFormat m_prev_format_options;
    OptionGroupReadMemory m_prev_memory_options;
    OptionGroupOutputFile m_prev_outfile_options;
    OptionGroupValueObjectDisplay m_prev_varobj_options;
    ClangASTType m_prev_clang_ast_type;
};


OptionDefinition
g_memory_write_option_table[] =
{
{ LLDB_OPT_SET_1, true,  "infile", 'i', required_argument, NULL, 0, eArgTypeFilename, "Write memory using the contents of a file."},
{ LLDB_OPT_SET_1, false, "offset", 'o', required_argument, NULL, 0, eArgTypeOffset,   "Start writng bytes from an offset within the input file."},
};


//----------------------------------------------------------------------
// Write memory to the inferior process
//----------------------------------------------------------------------
class CommandObjectMemoryWrite : public CommandObjectParsed
{
public:

    class OptionGroupWriteMemory : public OptionGroup
    {
    public:
        OptionGroupWriteMemory () :
            OptionGroup()
        {
        }

        virtual
        ~OptionGroupWriteMemory ()
        {
        }

        virtual uint32_t
        GetNumDefinitions ()
        {
            return sizeof (g_memory_write_option_table) / sizeof (OptionDefinition);
        }
        
        virtual const OptionDefinition*
        GetDefinitions ()
        {
            return g_memory_write_option_table;
        }
        
        virtual Error
        SetOptionValue (CommandInterpreter &interpreter,
                        uint32_t option_idx,
                        const char *option_arg)
        {
            Error error;
            const int short_option = g_memory_write_option_table[option_idx].short_option;
            
            switch (short_option)
            {
                case 'i':
                    m_infile.SetFile (option_arg, true);
                    if (!m_infile.Exists())
                    {
                        m_infile.Clear();
                        error.SetErrorStringWithFormat("input file does not exist: '%s'", option_arg);
                    }
                    break;
                    
                case 'o':
                    {
                        bool success;
                        m_infile_offset = Args::StringToUInt64(option_arg, 0, 0, &success);
                        if (!success)
                        {
                            error.SetErrorStringWithFormat("invalid offset string '%s'", option_arg);
                        }
                    }
                    break;
                    
                default:
                    error.SetErrorStringWithFormat("unrecognized short option '%c'", short_option);
                    break;
            }
            return error;
        }
        
        virtual void
        OptionParsingStarting (CommandInterpreter &interpreter)
        {
            m_infile.Clear();
            m_infile_offset = 0;
        }

        FileSpec m_infile;
        off_t m_infile_offset;
    };

    CommandObjectMemoryWrite (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "memory write",
                             "Write to the memory of the process being debugged.",
                             NULL,
                             eFlagRequiresProcess | eFlagProcessMustBeLaunched),
        m_option_group (interpreter),
        m_format_options (eFormatBytes, 1, UINT64_MAX),
        m_memory_options ()
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData addr_arg;
        CommandArgumentData value_arg;
        
        // Define the first (and only) variant of this arg.
        addr_arg.arg_type = eArgTypeAddress;
        addr_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (addr_arg);
        
        // Define the first (and only) variant of this arg.
        value_arg.arg_type = eArgTypeValue;
        value_arg.arg_repetition = eArgRepeatPlus;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg2.push_back (value_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
        
        m_option_group.Append (&m_format_options, OptionGroupFormat::OPTION_GROUP_FORMAT, LLDB_OPT_SET_1);
        m_option_group.Append (&m_format_options, OptionGroupFormat::OPTION_GROUP_SIZE  , LLDB_OPT_SET_1|LLDB_OPT_SET_2);
        m_option_group.Append (&m_memory_options, LLDB_OPT_SET_ALL, LLDB_OPT_SET_2);
        m_option_group.Finalize();

    }

    virtual
    ~CommandObjectMemoryWrite ()
    {
    }

    Options *
    GetOptions ()
    {
        return &m_option_group;
    }

    bool
    UIntValueIsValidForSize (uint64_t uval64, size_t total_byte_size)
    {
        if (total_byte_size > 8)
            return false;

        if (total_byte_size == 8)
            return true;

        const uint64_t max = ((uint64_t)1 << (uint64_t)(total_byte_size * 8)) - 1;
        return uval64 <= max;
    }

    bool
    SIntValueIsValidForSize (int64_t sval64, size_t total_byte_size)
    {
        if (total_byte_size > 8)
            return false;

        if (total_byte_size == 8)
            return true;

        const int64_t max = ((int64_t)1 << (uint64_t)(total_byte_size * 8 - 1)) - 1;
        const int64_t min = ~(max);
        return min <= sval64 && sval64 <= max;
    }

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        // No need to check "process" for validity as eFlagRequiresProcess ensures it is valid
        Process *process = m_exe_ctx.GetProcessPtr();

        const size_t argc = command.GetArgumentCount();

        if (m_memory_options.m_infile)
        {
            if (argc < 1)
            {
                result.AppendErrorWithFormat ("%s takes a destination address when writing file contents.\n", m_cmd_name.c_str());
                result.SetStatus(eReturnStatusFailed);
                return false;
            }       
        }
        else if (argc < 2)
        {
            result.AppendErrorWithFormat ("%s takes a destination address and at least one value.\n", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        StreamString buffer (Stream::eBinary,
                             process->GetTarget().GetArchitecture().GetAddressByteSize(),
                             process->GetTarget().GetArchitecture().GetByteOrder());

        OptionValueUInt64 &byte_size_value = m_format_options.GetByteSizeValue();
        size_t item_byte_size = byte_size_value.GetCurrentValue();

        Error error;
        lldb::addr_t addr = Args::StringToAddress (&m_exe_ctx,
                                                   command.GetArgumentAtIndex(0),
                                                   LLDB_INVALID_ADDRESS,
                                                   &error);

        if (addr == LLDB_INVALID_ADDRESS)
        {
            result.AppendError("invalid address expression\n");
            result.AppendError(error.AsCString());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        
        if (m_memory_options.m_infile)
        {
            size_t length = SIZE_MAX;
            if (item_byte_size > 0)
                length = item_byte_size;
            lldb::DataBufferSP data_sp (m_memory_options.m_infile.ReadFileContents (m_memory_options.m_infile_offset, length));
            if (data_sp)
            {
                length = data_sp->GetByteSize();
                if (length > 0)
                {
                    Error error;
                    size_t bytes_written = process->WriteMemory (addr, data_sp->GetBytes(), length, error);
                    
                    if (bytes_written == length)
                    {
                        // All bytes written
                        result.GetOutputStream().Printf("%" PRIu64 " bytes were written to 0x%" PRIx64 "\n", (uint64_t)bytes_written, addr);
                        result.SetStatus(eReturnStatusSuccessFinishResult);
                    }
                    else if (bytes_written > 0)
                    {
                        // Some byte written
                        result.GetOutputStream().Printf("%" PRIu64 " bytes of %" PRIu64 " requested were written to 0x%" PRIx64 "\n", (uint64_t)bytes_written, (uint64_t)length, addr);
                        result.SetStatus(eReturnStatusSuccessFinishResult);
                    }
                    else 
                    {
                        result.AppendErrorWithFormat ("Memory write to 0x%" PRIx64 " failed: %s.\n", addr, error.AsCString());
                        result.SetStatus(eReturnStatusFailed);
                    }
                }
            }
            else
            {
                result.AppendErrorWithFormat ("Unable to read contents of file.\n");
                result.SetStatus(eReturnStatusFailed);
            }
            return result.Succeeded();
        }
        else if (item_byte_size == 0)
        {
            if (m_format_options.GetFormat() == eFormatPointer)
                item_byte_size = buffer.GetAddressByteSize();
            else
                item_byte_size = 1;
        }

        command.Shift(); // shift off the address argument
        uint64_t uval64;
        int64_t sval64;
        bool success = false;
        const size_t num_value_args = command.GetArgumentCount();
        for (size_t i=0; i<num_value_args; ++i)
        {
            const char *value_str = command.GetArgumentAtIndex(i);

            switch (m_format_options.GetFormat())
            {
            case kNumFormats:
            case eFormatFloat:  // TODO: add support for floats soon
            case eFormatCharPrintable:
            case eFormatBytesWithASCII:
            case eFormatComplex:
            case eFormatEnum:
            case eFormatUnicode16:
            case eFormatUnicode32:
            case eFormatVectorOfChar:
            case eFormatVectorOfSInt8:
            case eFormatVectorOfUInt8:
            case eFormatVectorOfSInt16:
            case eFormatVectorOfUInt16:
            case eFormatVectorOfSInt32:
            case eFormatVectorOfUInt32:
            case eFormatVectorOfSInt64:
            case eFormatVectorOfUInt64:
            case eFormatVectorOfFloat32:
            case eFormatVectorOfFloat64:
            case eFormatVectorOfUInt128:
            case eFormatOSType:
            case eFormatComplexInteger:
            case eFormatAddressInfo:
            case eFormatHexFloat:
            case eFormatInstruction:
            case eFormatVoid:
                result.AppendError("unsupported format for writing memory");
                result.SetStatus(eReturnStatusFailed);
                return false;

            case eFormatDefault:
            case eFormatBytes:
            case eFormatHex:
            case eFormatHexUppercase:
            case eFormatPointer:
                
                // Decode hex bytes
                uval64 = Args::StringToUInt64(value_str, UINT64_MAX, 16, &success);
                if (!success)
                {
                    result.AppendErrorWithFormat ("'%s' is not a valid hex string value.\n", value_str);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                else if (!UIntValueIsValidForSize (uval64, item_byte_size))
                {
                    result.AppendErrorWithFormat ("Value 0x%" PRIx64 " is too large to fit in a %lu byte unsigned integer value.\n", uval64, item_byte_size);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                buffer.PutMaxHex64 (uval64, item_byte_size);
                break;

            case eFormatBoolean:
                uval64 = Args::StringToBoolean(value_str, false, &success);
                if (!success)
                {
                    result.AppendErrorWithFormat ("'%s' is not a valid boolean string value.\n", value_str);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                buffer.PutMaxHex64 (uval64, item_byte_size);
                break;

            case eFormatBinary:
                uval64 = Args::StringToUInt64(value_str, UINT64_MAX, 2, &success);
                if (!success)
                {
                    result.AppendErrorWithFormat ("'%s' is not a valid binary string value.\n", value_str);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                else if (!UIntValueIsValidForSize (uval64, item_byte_size))
                {
                    result.AppendErrorWithFormat ("Value 0x%" PRIx64 " is too large to fit in a %lu byte unsigned integer value.\n", uval64, item_byte_size);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                buffer.PutMaxHex64 (uval64, item_byte_size);
                break;

            case eFormatCharArray:
            case eFormatChar:
            case eFormatCString:
                if (value_str[0])
                {
                    size_t len = strlen (value_str);
                    // Include the NULL for C strings...
                    if (m_format_options.GetFormat() == eFormatCString)
                        ++len;
                    Error error;
                    if (process->WriteMemory (addr, value_str, len, error) == len)
                    {
                        addr += len;
                    }
                    else
                    {
                        result.AppendErrorWithFormat ("Memory write to 0x%" PRIx64 " failed: %s.\n", addr, error.AsCString());
                        result.SetStatus(eReturnStatusFailed);
                        return false;
                    }
                }
                break;

            case eFormatDecimal:
                sval64 = Args::StringToSInt64(value_str, INT64_MAX, 0, &success);
                if (!success)
                {
                    result.AppendErrorWithFormat ("'%s' is not a valid signed decimal value.\n", value_str);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                else if (!SIntValueIsValidForSize (sval64, item_byte_size))
                {
                    result.AppendErrorWithFormat ("Value %" PRIi64 " is too large or small to fit in a %lu byte signed integer value.\n", sval64, item_byte_size);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                buffer.PutMaxHex64 (sval64, item_byte_size);
                break;

            case eFormatUnsigned:
                uval64 = Args::StringToUInt64(value_str, UINT64_MAX, 0, &success);
                if (!success)
                {
                    result.AppendErrorWithFormat ("'%s' is not a valid unsigned decimal string value.\n", value_str);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                else if (!UIntValueIsValidForSize (uval64, item_byte_size))
                {
                    result.AppendErrorWithFormat ("Value %" PRIu64 " is too large to fit in a %lu byte unsigned integer value.\n", uval64, item_byte_size);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                buffer.PutMaxHex64 (uval64, item_byte_size);
                break;

            case eFormatOctal:
                uval64 = Args::StringToUInt64(value_str, UINT64_MAX, 8, &success);
                if (!success)
                {
                    result.AppendErrorWithFormat ("'%s' is not a valid octal string value.\n", value_str);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                else if (!UIntValueIsValidForSize (uval64, item_byte_size))
                {
                    result.AppendErrorWithFormat ("Value %" PRIo64 " is too large to fit in a %lu byte unsigned integer value.\n", uval64, item_byte_size);
                    result.SetStatus(eReturnStatusFailed);
                    return false;
                }
                buffer.PutMaxHex64 (uval64, item_byte_size);
                break;
            }
        }

        if (!buffer.GetString().empty())
        {
            Error error;
            if (process->WriteMemory (addr, buffer.GetString().c_str(), buffer.GetString().size(), error) == buffer.GetString().size())
                return true;
            else
            {
                result.AppendErrorWithFormat ("Memory write to 0x%" PRIx64 " failed: %s.\n", addr, error.AsCString());
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
        }
        return true;
    }

    OptionGroupOptions m_option_group;
    OptionGroupFormat m_format_options;
    OptionGroupWriteMemory m_memory_options;
};


//-------------------------------------------------------------------------
// CommandObjectMemory
//-------------------------------------------------------------------------

CommandObjectMemory::CommandObjectMemory (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "memory",
                            "A set of commands for operating on memory.",
                            "memory <subcommand> [<subcommand-options>]")
{
    LoadSubCommand ("read",  CommandObjectSP (new CommandObjectMemoryRead (interpreter)));
    LoadSubCommand ("write", CommandObjectSP (new CommandObjectMemoryWrite (interpreter)));
}

CommandObjectMemory::~CommandObjectMemory ()
{
}
