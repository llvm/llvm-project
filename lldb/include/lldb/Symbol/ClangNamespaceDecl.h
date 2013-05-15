//===-- ClangNamespaceDecl.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangNamespaceDecl_h_
#define liblldb_ClangNamespaceDecl_h_

#include "lldb/lldb-public.h"
#include "lldb/Core/ClangForward.h"

namespace lldb_private {
    
class ClangNamespaceDecl
{
public:
    ClangNamespaceDecl () :
        m_ast (NULL),
        m_namespace_decl (NULL)
    {
    }

    ClangNamespaceDecl (clang::ASTContext *ast, clang::NamespaceDecl *namespace_decl) :
        m_ast (ast),
        m_namespace_decl (namespace_decl)
    {
    }
    
    ClangNamespaceDecl (const ClangNamespaceDecl &rhs) :
        m_ast (rhs.m_ast),
        m_namespace_decl (rhs.m_namespace_decl)
    {
    }

    const ClangNamespaceDecl &
    operator = (const ClangNamespaceDecl &rhs)
    {
        m_ast = rhs.m_ast;
        m_namespace_decl = rhs.m_namespace_decl;
        return *this;
    }
    
    //------------------------------------------------------------------
    /// Convert to bool operator.
    ///
    /// This allows code to check a ClangNamespaceDecl object to see if 
    /// it contains a valid namespace decl using code such as:
    ///
    /// @code
    /// ClangNamespaceDecl ns_decl(...);
    /// if (ns_decl)
    /// { ...
    /// @endcode
    ///
    /// @return
    ///     /b True this object contains a valid namespace decl, \b 
    ///     false otherwise.
    //------------------------------------------------------------------
    operator bool() const
    {
        return m_ast != NULL && m_namespace_decl != NULL;
    }
    
    clang::ASTContext *
    GetASTContext() const
    { 
        return m_ast; 
    }

    void
    SetASTContext (clang::ASTContext *ast)
    { 
        m_ast = ast;
    }

    clang::NamespaceDecl *
    GetNamespaceDecl () const
    {
        return m_namespace_decl;
    }

    void
    SetNamespaceDecl (clang::NamespaceDecl *namespace_decl)
    {
        m_namespace_decl = namespace_decl;
    }

    std::string
    GetQualifiedName () const;

protected:
    clang::ASTContext  *m_ast;
    clang::NamespaceDecl *m_namespace_decl;
};
    

} // namespace lldb_private

#endif // #ifndef liblldb_ClangNamespaceDecl_h_
