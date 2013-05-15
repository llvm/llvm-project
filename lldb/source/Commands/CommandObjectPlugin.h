//===-- CommandObjectPlugin.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectPlugin_h_
#define liblldb_CommandObjectPlugin_h_

// C Includes
// C++ Includes


// Other libraries and framework includes
// Project includes

#include "lldb/lldb-types.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {
    
    class CommandObjectPlugin : public CommandObjectMultiword
    {
    public:
        CommandObjectPlugin (CommandInterpreter &interpreter);
        
        virtual
        ~CommandObjectPlugin ();
    };
    
} // namespace lldb_private

#endif  // liblldb_CommandObjectPlugin_h_
