//===-- CommandObjectSource.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "CommandObjectSource.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/FileLineResolver.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Interpreter/Options.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectSourceInfo
//-------------------------------------------------------------------------

class CommandObjectSourceInfo : public CommandObjectParsed
{

    class CommandOptions : public Options
    {
    public:
        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter)
        {
        }

        ~CommandOptions ()
        {
        }

        Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            const int short_option = g_option_table[option_idx].short_option;
            switch (short_option)
            {
            case 'l':
                start_line = Args::StringToUInt32 (option_arg, 0);
                if (start_line == 0)
                    error.SetErrorStringWithFormat("invalid line number: '%s'", option_arg);
                break;

             case 'f':
                file_name = option_arg;
                break;

           default:
                error.SetErrorStringWithFormat("unrecognized short option '%c'", short_option);
                break;
            }

            return error;
        }

        void
        OptionParsingStarting ()
        {
            file_spec.Clear();
            file_name.clear();
            start_line = 0;
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        FileSpec file_spec;
        std::string file_name;
        uint32_t start_line;
        
    };
 
public:   
    CommandObjectSourceInfo(CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "source info",
                             "Display information about the source lines from the current executable's debug info.",
                             "source info [<cmd-options>]"),
        m_options (interpreter)
    {
    }

    ~CommandObjectSourceInfo ()
    {
    }


    Options *
    GetOptions ()
    {
        return &m_options;
    }

protected:
    bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        result.AppendError ("Not yet implemented");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    CommandOptions m_options;
};

OptionDefinition
CommandObjectSourceInfo::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "line",       'l', required_argument, NULL, 0, eArgTypeLineNum,    "The line number at which to start the display source."},
{ LLDB_OPT_SET_1, false, "file",       'f', required_argument, NULL, CommandCompletions::eSourceFileCompletion, eArgTypeFilename,    "The file from which to display source."},
{ 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

#pragma mark CommandObjectSourceList
//-------------------------------------------------------------------------
// CommandObjectSourceList
//-------------------------------------------------------------------------

class CommandObjectSourceList : public CommandObjectParsed
{

    class CommandOptions : public Options
    {
    public:
        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter)
        {
        }

        ~CommandOptions ()
        {
        }

        Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            const int short_option = g_option_table[option_idx].short_option;
            switch (short_option)
            {
            case 'l':
                start_line = Args::StringToUInt32 (option_arg, 0);
                if (start_line == 0)
                    error.SetErrorStringWithFormat("invalid line number: '%s'", option_arg);
                break;

            case 'c':
                num_lines = Args::StringToUInt32 (option_arg, 0);
                if (num_lines == 0)
                    error.SetErrorStringWithFormat("invalid line count: '%s'", option_arg);
                break;

            case 'f':
                file_name = option_arg;
                break;
                
            case 'n':
                symbol_name = option_arg;
                break;

            case 'a':
                {
                    ExecutionContext exe_ctx (m_interpreter.GetExecutionContext());
                    address = Args::StringToAddress(&exe_ctx, option_arg, LLDB_INVALID_ADDRESS, &error);
                }
                break;
            case 's':
                modules.push_back (std::string (option_arg));
                break;
            
            case 'b':
                show_bp_locs = true;
                break;
            case 'r':
                reverse = true;
                break;
           default:
                error.SetErrorStringWithFormat("unrecognized short option '%c'", short_option);
                break;
            }

            return error;
        }

        void
        OptionParsingStarting ()
        {
            file_spec.Clear();
            file_name.clear();
            symbol_name.clear();
            address = LLDB_INVALID_ADDRESS;
            start_line = 0;
            num_lines = 0;
            show_bp_locs = false;
            reverse = false;
            modules.clear();
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        FileSpec file_spec;
        std::string file_name;
        std::string symbol_name;
        lldb::addr_t address;
        uint32_t start_line;
        uint32_t num_lines;
        STLStringArray modules;        
        bool show_bp_locs;
        bool reverse;
    };
 
public:   
    CommandObjectSourceList(CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "source list",
                             "Display source code (as specified) based on the current executable's debug info.",
                             NULL,
                             eFlagRequiresTarget), 
        m_options (interpreter)
    {
    }

    ~CommandObjectSourceList ()
    {
    }


    Options *
    GetOptions ()
    {
        return &m_options;
    }

    virtual const char *
    GetRepeatCommand (Args &current_command_args, uint32_t index)
    {
        // This is kind of gross, but the command hasn't been parsed yet so we can't look at the option
        // values for this invocation...  I have to scan the arguments directly.
        size_t num_args = current_command_args.GetArgumentCount();
        bool is_reverse = false;
        for (size_t i = 0 ; i < num_args; i++)
        {
            const char *arg = current_command_args.GetArgumentAtIndex(i);
            if (arg && (strcmp(arg, "-r") == 0 || strcmp(arg, "--reverse") == 0))
            {
                is_reverse = true;
            }
        }
        if (is_reverse)
        {
            if (m_reverse_name.empty())
            {
                m_reverse_name = m_cmd_name;
                m_reverse_name.append (" -r");
            }
            return m_reverse_name.c_str();
        }
        else
            return m_cmd_name.c_str();
    }

protected:
    bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();

        if (argc != 0)
        {
            result.AppendErrorWithFormat("'%s' takes no arguments, only flags.\n", GetCommandName());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        Target *target = m_exe_ctx.GetTargetPtr();

        SymbolContextList sc_list;
        if (!m_options.symbol_name.empty())
        {
            // Displaying the source for a symbol:
            ConstString name(m_options.symbol_name.c_str());
            bool include_symbols = false;
            bool include_inlines = true;
            bool append = true;
            size_t num_matches = 0;
            
            if (m_options.num_lines == 0)
                m_options.num_lines = 10;

            const size_t num_modules = m_options.modules.size();
            if (num_modules > 0)
            {
                ModuleList matching_modules;
                for (size_t i = 0; i < num_modules; ++i)
                {
                    FileSpec module_file_spec(m_options.modules[i].c_str(), false);
                    if (module_file_spec)
                    {
                        ModuleSpec module_spec (module_file_spec);
                        matching_modules.Clear();
                        target->GetImages().FindModules (module_spec, matching_modules);
                        num_matches += matching_modules.FindFunctions (name, eFunctionNameTypeAuto, include_symbols, include_inlines, append, sc_list);
                    }
                }
            }
            else
            {
                num_matches = target->GetImages().FindFunctions (name, eFunctionNameTypeAuto, include_symbols, include_inlines, append, sc_list);
            }
            
            SymbolContext sc;

            if (num_matches == 0)
            {
                result.AppendErrorWithFormat("Could not find function named: \"%s\".\n", m_options.symbol_name.c_str());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            
            sc_list.GetContextAtIndex (0, sc);
            FileSpec start_file;
            uint32_t start_line;
            uint32_t end_line;
            FileSpec end_file;
            if (sc.function != NULL)
            {
                sc.function->GetStartLineSourceInfo (start_file, start_line);
                if (start_line == 0)
                {
                    result.AppendErrorWithFormat("Could not find line information for start of function: \"%s\".\n", m_options.symbol_name.c_str());
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                sc.function->GetEndLineSourceInfo (end_file, end_line);
            }
            else
            {
                result.AppendErrorWithFormat("Could not find function info for: \"%s\".\n", m_options.symbol_name.c_str());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }

            if (num_matches > 1)
            {
                // This could either be because there are multiple functions of this name, in which case
                // we'll have to specify this further...  Or it could be because there are multiple inlined instances
                // of one function.  So run through the matches and if they all have the same file & line then we can just
                // list one.
                
                bool found_multiple = false;
                
                for (size_t i = 1; i < num_matches; i++)
                {
                    SymbolContext scratch_sc;
                    sc_list.GetContextAtIndex (i, scratch_sc);
                    if (scratch_sc.function != NULL)
                    {
                        FileSpec scratch_file;
                        uint32_t scratch_line;
                        scratch_sc.function->GetStartLineSourceInfo (scratch_file, scratch_line);
                        if (scratch_file != start_file 
                            || scratch_line != start_line)
                        {
                            found_multiple = true;
                            break;
                        }
                    }
                }
                if (found_multiple)
                {
                    StreamString s;
                    for (size_t i = 0; i < num_matches; i++)
                    {
                        SymbolContext scratch_sc;
                        sc_list.GetContextAtIndex (i, scratch_sc);
                        if (scratch_sc.function != NULL)
                        {
                            s.Printf("\n%lu: ", i); 
                            scratch_sc.function->Dump (&s, true);
                        }
                    }
                    result.AppendErrorWithFormat("Multiple functions found matching: %s: \n%s\n", 
                                                 m_options.symbol_name.c_str(),
                                                 s.GetData());
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            
            // This is a little hacky, but the first line table entry for a function points to the "{" that 
            // starts the function block.  It would be nice to actually get the function
            // declaration in there too.  So back up a bit, but not further than what you're going to display.
            uint32_t extra_lines;
            if (m_options.num_lines >= 10)
                extra_lines = 5;
            else
                extra_lines = m_options.num_lines/2;
            uint32_t line_no;
            if (start_line <= extra_lines)
                line_no = 1;
            else
                line_no = start_line - extra_lines;
                
            // For fun, if the function is shorter than the number of lines we're supposed to display, 
            // only display the function...
            if (end_line != 0)
            {
                if (m_options.num_lines > end_line - line_no)
                    m_options.num_lines = end_line - line_no + extra_lines;
            }
            
            char path_buf[PATH_MAX];
            start_file.GetPath(path_buf, sizeof(path_buf));
            
            if (m_options.show_bp_locs)
            {
                const bool show_inlines = true;
                m_breakpoint_locations.Reset (start_file, 0, show_inlines);
                SearchFilter target_search_filter (m_exe_ctx.GetTargetSP());
                target_search_filter.Search (m_breakpoint_locations);
            }
            else
                m_breakpoint_locations.Clear();

            result.AppendMessageWithFormat("File: %s\n", path_buf);
            target->GetSourceManager().DisplaySourceLinesWithLineNumbers (start_file,
                                                                          line_no,
                                                                          0,
                                                                          m_options.num_lines,
                                                                          "",
                                                                          &result.GetOutputStream(),
                                                                          GetBreakpointLocations ());
            
            result.SetStatus (eReturnStatusSuccessFinishResult);
            return true;

        }
        else if (m_options.address != LLDB_INVALID_ADDRESS)
        {
            SymbolContext sc;
            Address so_addr;
            StreamString error_strm;

            if (target->GetSectionLoadList().IsEmpty())
            {
                // The target isn't loaded yet, we need to lookup the file address
                // in all modules
                const ModuleList &module_list = target->GetImages();
                const size_t num_modules = module_list.GetSize();
                for (size_t i=0; i<num_modules; ++i)
                {
                    ModuleSP module_sp (module_list.GetModuleAtIndex(i));
                    if (module_sp && module_sp->ResolveFileAddress(m_options.address, so_addr))
                    {
                        sc.Clear(true);
                        if (module_sp->ResolveSymbolContextForAddress (so_addr, eSymbolContextEverything, sc) & eSymbolContextLineEntry)
                            sc_list.Append(sc);
                    }
                }
                
                if (sc_list.GetSize() == 0)
                {
                    result.AppendErrorWithFormat("no modules have source information for file address 0x%" PRIx64 ".\n",
                                                 m_options.address);
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            else
            {
                // The target has some things loaded, resolve this address to a
                // compile unit + file + line and display
                if (target->GetSectionLoadList().ResolveLoadAddress (m_options.address, so_addr))
                {
                    ModuleSP module_sp (so_addr.GetModule());
                    if (module_sp)
                    {
                        sc.Clear(true);
                        if (module_sp->ResolveSymbolContextForAddress (so_addr, eSymbolContextEverything, sc) & eSymbolContextLineEntry)
                        {
                            sc_list.Append(sc);
                        }
                        else
                        {
                            so_addr.Dump(&error_strm, NULL, Address::DumpStyleModuleWithFileAddress);
                            result.AppendErrorWithFormat("address resolves to %s, but there is no line table information available for this address.\n",
                                                         error_strm.GetData());
                            result.SetStatus (eReturnStatusFailed);
                            return false;
                        }
                    }
                }

                if (sc_list.GetSize() == 0)
                {
                    result.AppendErrorWithFormat("no modules contain load address 0x%" PRIx64 ".\n", m_options.address);
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            uint32_t num_matches = sc_list.GetSize();
            for (uint32_t i=0; i<num_matches; ++i)
            {
                sc_list.GetContextAtIndex(i, sc);
                if (sc.comp_unit)
                {
                    if (m_options.show_bp_locs)
                    {
                        m_breakpoint_locations.Clear();
                        const bool show_inlines = true;
                        m_breakpoint_locations.Reset (*sc.comp_unit, 0, show_inlines);
                        SearchFilter target_search_filter (target->shared_from_this());
                        target_search_filter.Search (m_breakpoint_locations);
                    }
                    
                    bool show_fullpaths = true;
                    bool show_module = true;
                    bool show_inlined_frames = true;
                    sc.DumpStopContext(&result.GetOutputStream(),
                                       m_exe_ctx.GetBestExecutionContextScope(),
                                       sc.line_entry.range.GetBaseAddress(),
                                       show_fullpaths,
                                       show_module,
                                       show_inlined_frames);
                    result.GetOutputStream().EOL();

                    if (m_options.num_lines == 0)
                        m_options.num_lines = 10;
                    
                    size_t lines_to_back_up = m_options.num_lines >= 10 ? 5 : m_options.num_lines/2;

                    target->GetSourceManager().DisplaySourceLinesWithLineNumbers (sc.comp_unit,
                                                                                  sc.line_entry.line,
                                                                                  lines_to_back_up,
                                                                                  m_options.num_lines - lines_to_back_up,
                                                                                  "->",
                                                                                  &result.GetOutputStream(),
                                                                                  GetBreakpointLocations ());
                    result.SetStatus (eReturnStatusSuccessFinishResult);
                }
            }
        }
        else if (m_options.file_name.empty())
        {
            // Last valid source manager context, or the current frame if no
            // valid last context in source manager.
            // One little trick here, if you type the exact same list command twice in a row, it is
            // more likely because you typed it once, then typed it again
            if (m_options.start_line == 0)
            {
                if (target->GetSourceManager().DisplayMoreWithLineNumbers (&result.GetOutputStream(),
                                                                           m_options.num_lines,
                                                                           m_options.reverse,
                                                                           GetBreakpointLocations ()))
                {
                    result.SetStatus (eReturnStatusSuccessFinishResult);
                }
            }
            else
            {
                if (m_options.num_lines == 0)
                    m_options.num_lines = 10;

                if (m_options.show_bp_locs)
                {
                    SourceManager::FileSP last_file_sp (target->GetSourceManager().GetLastFile ());
                    if (last_file_sp)
                    {
                        const bool show_inlines = true;
                        m_breakpoint_locations.Reset (last_file_sp->GetFileSpec(), 0, show_inlines);
                        SearchFilter target_search_filter (target->shared_from_this());
                        target_search_filter.Search (m_breakpoint_locations);
                    }
                }
                else
                    m_breakpoint_locations.Clear();

                if (target->GetSourceManager().DisplaySourceLinesWithLineNumbersUsingLastFile(
                            m_options.start_line,   // Line to display
                            m_options.num_lines,    // Lines after line to
                            UINT32_MAX,             // Don't mark "line"
                            "",                     // Don't mark "line"
                            &result.GetOutputStream(),
                            GetBreakpointLocations ()))
                {
                    result.SetStatus (eReturnStatusSuccessFinishResult);
                }

            }
        }
        else
        {
            const char *filename = m_options.file_name.c_str();

            bool check_inlines = false;
            SymbolContextList sc_list;
            size_t num_matches = 0;
            
            if (m_options.modules.size() > 0)
            {
                ModuleList matching_modules;
                for (size_t i = 0, e = m_options.modules.size(); i < e; ++i)
                {
                    FileSpec module_file_spec(m_options.modules[i].c_str(), false);
                    if (module_file_spec)
                    {
                        ModuleSpec module_spec (module_file_spec);
                        matching_modules.Clear();
                        target->GetImages().FindModules (module_spec, matching_modules);
                        num_matches += matching_modules.ResolveSymbolContextForFilePath (filename,
                                                                                         0,
                                                                                         check_inlines,
                                                                                         eSymbolContextModule | eSymbolContextCompUnit,
                                                                                         sc_list);
                    }
                }
            }
            else
            {
                num_matches = target->GetImages().ResolveSymbolContextForFilePath (filename,
                                                                                   0,
                                                                                   check_inlines,
                                                                                   eSymbolContextModule | eSymbolContextCompUnit,
                                                                                   sc_list);
            }
            
            if (num_matches == 0)
            {
                result.AppendErrorWithFormat("Could not find source file \"%s\".\n", 
                                             m_options.file_name.c_str());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            
            if (num_matches > 1)
            {
                SymbolContext sc;
                bool got_multiple = false;
                FileSpec *test_cu_spec = NULL;

                for (unsigned i = 0; i < num_matches; i++)
                {
                    sc_list.GetContextAtIndex(i, sc);
                    if (sc.comp_unit)
                    {
                        if (test_cu_spec)
                        {
                            if (test_cu_spec != static_cast<FileSpec *> (sc.comp_unit))
                                got_multiple = true;
                                break;
                        }
                        else
                            test_cu_spec = sc.comp_unit;
                    }
                }
                if (got_multiple)
                {
                    result.AppendErrorWithFormat("Multiple source files found matching: \"%s.\"\n", 
                                                 m_options.file_name.c_str());
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            
            SymbolContext sc;
            if (sc_list.GetContextAtIndex(0, sc))
            {
                if (sc.comp_unit)
                {
                    if (m_options.show_bp_locs)
                    {
                        const bool show_inlines = true;
                        m_breakpoint_locations.Reset (*sc.comp_unit, 0, show_inlines);
                        SearchFilter target_search_filter (target->shared_from_this());
                        target_search_filter.Search (m_breakpoint_locations);
                    }
                    else
                        m_breakpoint_locations.Clear();

                    if (m_options.num_lines == 0)
                        m_options.num_lines = 10;
                    
                    target->GetSourceManager().DisplaySourceLinesWithLineNumbers (sc.comp_unit,
                                                                                  m_options.start_line,
                                                                                  0,
                                                                                  m_options.num_lines,
                                                                                  "",
                                                                                  &result.GetOutputStream(),
                                                                                  GetBreakpointLocations ());

                    result.SetStatus (eReturnStatusSuccessFinishResult);
                }
                else
                {
                    result.AppendErrorWithFormat("No comp unit found for: \"%s.\"\n", 
                                                 m_options.file_name.c_str());
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
        }
        return result.Succeeded();
    }
    
    const SymbolContextList *
    GetBreakpointLocations ()
    {
        if (m_breakpoint_locations.GetFileLineMatches().GetSize() > 0)
            return &m_breakpoint_locations.GetFileLineMatches();
        return NULL;
    }
    CommandOptions m_options;
    FileLineResolver m_breakpoint_locations;
    std::string    m_reverse_name;

};

OptionDefinition
CommandObjectSourceList::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_ALL, false, "count",  'c', required_argument, NULL, 0, eArgTypeCount,   "The number of source lines to display."},
{ LLDB_OPT_SET_1  |
  LLDB_OPT_SET_2  , false, "shlib",  's', required_argument, NULL, CommandCompletions::eModuleCompletion, eArgTypeShlibName, "Look up the source file in the given shared library."},
{ LLDB_OPT_SET_ALL, false, "show-breakpoints", 'b', no_argument, NULL, 0, eArgTypeNone, "Show the line table locations from the debug information that indicate valid places to set source level breakpoints."},
{ LLDB_OPT_SET_1  , false, "file",   'f', required_argument, NULL, CommandCompletions::eSourceFileCompletion, eArgTypeFilename,    "The file from which to display source."},
{ LLDB_OPT_SET_1  , false, "line",   'l', required_argument, NULL, 0, eArgTypeLineNum,    "The line number at which to start the display source."},
{ LLDB_OPT_SET_2  , false, "name",   'n', required_argument, NULL, CommandCompletions::eSymbolCompletion, eArgTypeSymbol,    "The name of a function whose source to display."},
{ LLDB_OPT_SET_3  , false, "address",'a', required_argument, NULL, 0, eArgTypeAddressOrExpression, "Lookup the address and display the source information for the corresponding file and line."},
{ LLDB_OPT_SET_4, false, "reverse", 'r', no_argument, NULL, 0, eArgTypeNone, "Reverse the listing to look backwards from the last displayed block of source."},
{ 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

#pragma mark CommandObjectMultiwordSource

//-------------------------------------------------------------------------
// CommandObjectMultiwordSource
//-------------------------------------------------------------------------

CommandObjectMultiwordSource::CommandObjectMultiwordSource (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "source",
                            "A set of commands for accessing source file information",
                            "source <subcommand> [<subcommand-options>]")
{
    // "source info" isn't implemented yet...
    //LoadSubCommand ("info",   CommandObjectSP (new CommandObjectSourceInfo (interpreter)));
    LoadSubCommand ("list",   CommandObjectSP (new CommandObjectSourceList (interpreter)));
}

CommandObjectMultiwordSource::~CommandObjectMultiwordSource ()
{
}

