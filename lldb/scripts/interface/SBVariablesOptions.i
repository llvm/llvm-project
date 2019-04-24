//===-- SWIG Interface for SBVariablesOptions ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

    explicit operator bool() const;
    
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
