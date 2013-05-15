//===-- CommandObjectTarget.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectTarget_h_
#define liblldb_CommandObjectTarget_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectMultiwordTarget
//-------------------------------------------------------------------------

class CommandObjectMultiwordTarget : public CommandObjectMultiword
{
public:

    CommandObjectMultiwordTarget (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectMultiwordTarget ();


};

} // namespace lldb_private

#endif  // liblldb_CommandObjectTarget_h_
