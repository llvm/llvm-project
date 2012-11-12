//===-- CommandObjectWatchpointCommand.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectWatchpointCommand_h_
#define liblldb_CommandObjectWatchpointCommand_h_

// C Includes
// C++ Includes


// Other libraries and framework includes
// Project includes

#include "lldb/lldb-types.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"


namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectMultiwordWatchpoint
//-------------------------------------------------------------------------

class CommandObjectWatchpointCommand : public CommandObjectMultiword
{
public:
    CommandObjectWatchpointCommand (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectWatchpointCommand ();

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectWatchpointCommand_h_
