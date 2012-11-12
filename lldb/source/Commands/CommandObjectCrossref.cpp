//===-- CommandObjectCrossref.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandObjectCrossref.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectCrossref
//-------------------------------------------------------------------------

CommandObjectCrossref::CommandObjectCrossref
(
    CommandInterpreter &interpreter,
    const char *name,
    const char *help,
    const char *syntax
) :
    CommandObjectParsed (interpreter, name, help, syntax),
    m_crossref_object_types()
{
}

CommandObjectCrossref::~CommandObjectCrossref ()
{
}

bool
CommandObjectCrossref::DoExecute (Args& command, CommandReturnObject &result)
{
    if (m_crossref_object_types.GetArgumentCount() == 0)
    {
        result.AppendErrorWithFormat ("There are no objects for which you can call '%s'.\n", GetCommandName());
        result.SetStatus (eReturnStatusFailed);
    }
    else
    {
        GenerateHelpText (result);
    }
    return result.Succeeded();
}

void
CommandObjectCrossref::AddObject (const char *obj_name)
{
    m_crossref_object_types.AppendArgument (obj_name);
}

const char **
CommandObjectCrossref::GetObjectTypes () const
{
    return m_crossref_object_types.GetConstArgumentVector();
}

void
CommandObjectCrossref::GenerateHelpText (CommandReturnObject &result)
{
    result.AppendMessage ("This command can be called on the following types of objects:");

    const size_t count = m_crossref_object_types.GetArgumentCount();
    for (size_t i = 0; i < count; ++i)
    {
        const char *obj_name = m_crossref_object_types.GetArgumentAtIndex(i);
        result.AppendMessageWithFormat ("    %s    (e.g.  '%s %s')\n", obj_name,
                                        obj_name, GetCommandName());
    }

    result.SetStatus (eReturnStatusSuccessFinishNoResult);
}

bool
CommandObjectCrossref::IsCrossRefObject ()
{
    return true;
}
