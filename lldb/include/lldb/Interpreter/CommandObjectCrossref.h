//===-- CommandObjectCrossref.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectCrossref_h_
#define liblldb_CommandObjectCrossref_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/Args.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectCrossref
//-------------------------------------------------------------------------

class CommandObjectCrossref : public CommandObjectParsed
{
public:
    CommandObjectCrossref (CommandInterpreter &interpreter,
                           const char *name,
                           const char *help = NULL,
                           const char *syntax = NULL);
    
    virtual
    ~CommandObjectCrossref ();

    virtual void
    GenerateHelpText (CommandReturnObject &result);

    virtual bool
    IsCrossRefObject ();

    virtual void
    AddObject (const char *obj_name);

    const char **
    GetObjectTypes () const;

protected:
    virtual bool
    DoExecute (Args& command,
             CommandReturnObject &result);

private:
    Args m_crossref_object_types;
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectCrossref_h_
