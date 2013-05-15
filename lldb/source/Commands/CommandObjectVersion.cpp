//===-- CommandObjectVersion.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "CommandObjectVersion.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectVersion
//-------------------------------------------------------------------------

CommandObjectVersion::CommandObjectVersion (CommandInterpreter &interpreter) :
    CommandObjectParsed (interpreter, "version", "Show version of LLDB debugger.", "version")
{
}

CommandObjectVersion::~CommandObjectVersion ()
{
}

bool
CommandObjectVersion::DoExecute (Args& args, CommandReturnObject &result)
{
    result.AppendMessageWithFormat ("%s\n", lldb_private::GetVersion());
    result.SetStatus (eReturnStatusSuccessFinishResult);
    return true;
}

