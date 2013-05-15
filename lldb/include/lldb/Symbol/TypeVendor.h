//===-- TypeVendor.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_TypeVendor_h_
#define liblldb_TypeVendor_h_

#include "lldb/Core/ClangForward.h"

namespace lldb_private {
    
//----------------------------------------------------------------------
// The type vendor class is intended as a generic interface to search
// for Clang types that are not necessarily backed by a specific symbol
// file.
//----------------------------------------------------------------------
class TypeVendor
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    TypeVendor()
    {
    }
    
    virtual
    ~TypeVendor()
    {
    }
    
    virtual uint32_t
    FindTypes (const ConstString &name,
               bool append,
               uint32_t max_matches,
               std::vector <ClangASTType> &types) = 0;
    
    virtual clang::ASTContext *
    GetClangASTContext () = 0;

protected:
    //------------------------------------------------------------------
    // Classes that inherit from TypeVendor can see and modify these
    //------------------------------------------------------------------
    
private:
    //------------------------------------------------------------------
    // For TypeVendor only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (TypeVendor);
};
    
    
} // namespace lldb_private

#endif
