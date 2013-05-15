//===-- DynamicLibrary.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DynamicLibrary_h_
#define liblldb_DynamicLibrary_h_

#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"

namespace lldb_private {

class DynamicLibrary
{
public:
    DynamicLibrary (const FileSpec& spec, uint32_t options = Host::eDynamicLibraryOpenOptionLazy |
                                                             Host::eDynamicLibraryOpenOptionLocal |
                                                             Host::eDynamicLibraryOpenOptionLimitGetSymbol);
    
    ~DynamicLibrary ();
    
    template <typename T = void*>
    T GetSymbol (const char* name)
    {
        Error err;
        if (!m_handle)
            return (T)NULL;
        void* symbol = Host::DynamicLibraryGetSymbol (m_handle, name, err);
        if (!symbol)
            return (T)NULL;
        return (T)symbol;
    }
    
    bool
    IsValid ();
    
private:
    lldb_private::FileSpec m_filespec;
    void* m_handle;
    
    DISALLOW_COPY_AND_ASSIGN (DynamicLibrary);
};
    
} // namespace lldb_private

#endif  // liblldb_DynamicLibrary_h_
