//===-- SWIG Interface for SBThreadCollection -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

namespace lldb {

%feature("docstring",
"Represents a collection of SBThread objects."
) SBThreadCollection;
class SBThreadCollection
{
public:
    
    SBThreadCollection ();
    
    SBThreadCollection (const SBThreadCollection &rhs);
    
    ~SBThreadCollection ();
    
    bool
    IsValid () const;
    
    size_t
    GetSize ();
    
    lldb::SBThread
    GetThreadAtIndex (size_t idx);
    
};
    
} // namespace lldb
