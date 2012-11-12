//===-- CommandObjectBreakpoint.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectBreakpoint_h_
#define liblldb_CommandObjectBreakpoint_h_

// C Includes
// C++ Includes

#include <utility>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Address.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/STLUtils.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectMultiwordBreakpoint
//-------------------------------------------------------------------------

class CommandObjectMultiwordBreakpoint : public CommandObjectMultiword
{
public:
    CommandObjectMultiwordBreakpoint (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectMultiwordBreakpoint ();

    static void
    VerifyBreakpointIDs (Args &args, Target *target, CommandReturnObject &result, BreakpointIDList *valid_ids);

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectBreakpoint_h_
