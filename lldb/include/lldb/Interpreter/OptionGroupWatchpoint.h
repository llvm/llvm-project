//===-- OptionGroupWatchpoint.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionGroupWatchpoint_h_
#define liblldb_OptionGroupWatchpoint_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// OptionGroupWatchpoint
//-------------------------------------------------------------------------

    class OptionGroupWatchpoint : public OptionGroup
    {
    public:
        
        static bool
        IsWatchSizeSupported(uint32_t watch_size);

        OptionGroupWatchpoint ();

        virtual
        ~OptionGroupWatchpoint ();
        
        virtual uint32_t
        GetNumDefinitions ();
        
        virtual const OptionDefinition*
        GetDefinitions ();
        
        virtual Error
        SetOptionValue (CommandInterpreter &interpreter,
                        uint32_t option_idx, 
                        const char *option_arg);
        
        virtual void
        OptionParsingStarting (CommandInterpreter &interpreter);
        
        // Note:
        // eWatchRead == LLDB_WATCH_TYPE_READ; and
        // eWatchWrite == LLDB_WATCH_TYPE_WRITE
        typedef enum WatchType {
            eWatchInvalid = 0,
            eWatchRead,
            eWatchWrite,
            eWatchReadWrite
        } WatchType;

        WatchType watch_type;
        uint32_t watch_size;
        bool watch_type_specified;

    private:
        DISALLOW_COPY_AND_ASSIGN(OptionGroupWatchpoint);
    };
    
} // namespace lldb_private

#endif  // liblldb_OptionGroupWatchpoint_h_
