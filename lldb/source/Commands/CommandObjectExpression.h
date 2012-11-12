//===-- CommandObjectExpression.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectExpression_h_
#define liblldb_CommandObjectExpression_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/OptionGroupFormat.h"
#include "lldb/Target/ExecutionContext.h"

namespace lldb_private {

class CommandObjectExpression : public CommandObjectRaw
{
public:

    class CommandOptions : public OptionGroup
    {
    public:

        CommandOptions ();

        virtual
        ~CommandOptions ();

        virtual uint32_t
        GetNumDefinitions ();
        
        virtual const OptionDefinition*
        GetDefinitions ();
        
        virtual Error
        SetOptionValue (CommandInterpreter &interpreter,
                        uint32_t option_idx,
                        const char *option_value);
        
        virtual void
        OptionParsingStarting (CommandInterpreter &interpreter);

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];
        bool        print_object;
        LazyBool    use_dynamic;
        bool        unwind_on_error;
        bool        show_types;
        bool        show_summary;
        uint32_t    timeout;
        bool        try_all_threads;
    };

    CommandObjectExpression (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectExpression ();

    virtual
    Options *
    GetOptions ();

protected:
    virtual bool
    DoExecute (const char *command,
               CommandReturnObject &result);

    static size_t
    MultiLineExpressionCallback (void *baton, 
                                 InputReader &reader, 
                                 lldb::InputReaderAction notification,
                                 const char *bytes, 
                                 size_t bytes_len);

    bool
    EvaluateExpression (const char *expr,
                        Stream *output_stream,
                        Stream *error_stream,
                        CommandReturnObject *result = NULL);

    OptionGroupOptions m_option_group;
    OptionGroupFormat m_format_options;
    CommandOptions m_command_options;
    uint32_t m_expr_line_count;
    std::string m_expr_lines; // Multi-line expression support
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectExpression_h_
