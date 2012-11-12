//===-- CommandObjectQuit.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectQuit.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectQuit
//-------------------------------------------------------------------------

CommandObjectQuit::CommandObjectQuit (CommandInterpreter &interpreter) :
    CommandObjectParsed (interpreter, "quit", "Quit out of the LLDB debugger.", "quit")
{
}

CommandObjectQuit::~CommandObjectQuit ()
{
}

bool
CommandObjectQuit::DoExecute (Args& command, CommandReturnObject &result)
{
    m_interpreter.BroadcastEvent (CommandInterpreter::eBroadcastBitQuitCommandReceived);
    result.SetStatus (eReturnStatusQuit);
    return true;
}

