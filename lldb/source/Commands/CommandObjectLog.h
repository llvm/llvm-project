//===-- CommandObjectLog.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectLog_h_
#define liblldb_CommandObjectLog_h_

// C Includes
// C++ Includes
#include <map>
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectLog
//-------------------------------------------------------------------------

class CommandObjectLog : public CommandObjectMultiword
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    CommandObjectLog(CommandInterpreter &interpreter);

    virtual
    ~CommandObjectLog();

private:
    //------------------------------------------------------------------
    // For CommandObjectLog only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (CommandObjectLog);
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectLog_h_
