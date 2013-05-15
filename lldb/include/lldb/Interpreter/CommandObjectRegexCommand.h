//===-- CommandObjectRegexCommand.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectRegexCommand_h_
#define liblldb_CommandObjectRegexCommand_h_

// C Includes
// C++ Includes
#include <list>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/RegularExpression.h"
#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectRegexCommand
//-------------------------------------------------------------------------

class CommandObjectRegexCommand : public CommandObjectRaw
{
public:

    CommandObjectRegexCommand (CommandInterpreter &interpreter,
                               const char *name, 
                               const char *help, 
                               const char *syntax, 
                               uint32_t max_matches,
                               uint32_t completion_type_mask = 0);
    
    virtual
    ~CommandObjectRegexCommand ();

    bool
    AddRegexCommand (const char *re_cstr, const char *command_cstr);

    bool
    HasRegexEntries () const
    {
        return !m_entries.empty();
    }
    
    virtual int
    HandleCompletion (Args &input,
                      int &cursor_index,
                      int &cursor_char_position,
                      int match_start_point,
                      int max_return_elements,
                      bool &word_complete,
                      StringList &matches);

protected:
    virtual bool
    DoExecute (const char *command, CommandReturnObject &result);

    struct Entry
    {
        RegularExpression regex;
        std::string command;
    };

    typedef std::list<Entry> EntryCollection;
    const uint32_t m_max_matches;
    const uint32_t m_completion_type_mask;
    EntryCollection m_entries;

private:
    DISALLOW_COPY_AND_ASSIGN (CommandObjectRegexCommand);
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectRegexCommand_h_
