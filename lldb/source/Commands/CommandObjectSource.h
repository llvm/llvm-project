//===-- CommandObjectSource.h.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectSource_h_
#define liblldb_CommandObjectSource_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Core/STLUtils.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectMultiwordSource
//-------------------------------------------------------------------------

class CommandObjectMultiwordSource : public CommandObjectMultiword
{
public:

    CommandObjectMultiwordSource (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectMultiwordSource ();

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectSource.h_h_
