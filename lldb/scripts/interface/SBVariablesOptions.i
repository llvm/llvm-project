//===-- SWIG Interface for SBVariablesOptions ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {
    
class SBVariablesOptions
{
public:
    SBVariablesOptions ();
    
    SBVariablesOptions (const SBVariablesOptions& options);
    
    ~SBVariablesOptions ();
    
    bool
    IsValid () const;
    
    bool
    GetIncludeArguments ()  const;
    
    void
    SetIncludeArguments (bool);

    bool
    GetIncludeRecognizedArguments (const lldb::SBTarget &)  const;

    void
    SetIncludeRecognizedArguments (bool);

    bool
    GetIncludeLocals ()  const;
    
    void
    SetIncludeLocals (bool);
    
    bool
    GetIncludeStatics ()  const;
    
    void
    SetIncludeStatics (bool);
    
    bool
    GetInScopeOnly ()  const;
    
    void
    SetInScopeOnly (bool);
    
    bool
    GetIncludeRuntimeSupportValues () const;
    
    void
    SetIncludeRuntimeSupportValues (bool);
    
    lldb::DynamicValueType
    GetUseDynamic () const;
    
    void
    SetUseDynamic (lldb::DynamicValueType);
};
    
} // namespace lldb
