//===-- CommandObjectCommands.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectCommands_h_
#define liblldb_CommandObjectCommands_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Core/STLUtils.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectMultiwordCommands
//-------------------------------------------------------------------------

class CommandObjectMultiwordCommands : public CommandObjectMultiword
{
public:

    CommandObjectMultiwordCommands (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectMultiwordCommands ();

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectCommands_h_
