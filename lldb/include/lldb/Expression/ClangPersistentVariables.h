//===-- ClangPersistentVariables.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangPersistentVariables_h_
#define liblldb_ClangPersistentVariables_h_

#include "lldb/Expression/ClangExpressionVariable.h"
#include "llvm/ADT/DenseMap.h"

namespace lldb_private
{
    
//----------------------------------------------------------------------
/// @class ClangPersistentVariables ClangPersistentVariables.h "lldb/Expression/ClangPersistentVariables.h"
/// @brief Manages persistent values that need to be preserved between expression invocations.
///
/// A list of variables that can be accessed and updated by any expression.  See
/// ClangPersistentVariable for more discussion.  Also provides an increasing,
/// 0-based counter for naming result variables.
//----------------------------------------------------------------------
class ClangPersistentVariables : public ClangExpressionVariableList
{
public:
    
    //----------------------------------------------------------------------
    /// Constructor
    //----------------------------------------------------------------------
    ClangPersistentVariables ();

    lldb::ClangExpressionVariableSP
    CreatePersistentVariable (const lldb::ValueObjectSP &valobj_sp);

    lldb::ClangExpressionVariableSP
    CreatePersistentVariable (ExecutionContextScope *exe_scope,
                              const ConstString &name, 
                              const TypeFromUser& user_type, 
                              lldb::ByteOrder byte_order, 
                              uint32_t addr_byte_size);

    //----------------------------------------------------------------------
    /// Return the next entry in the sequence of strings "$0", "$1", ... for
    /// use naming persistent expression convenience variables.
    ///
    /// @return
    ///     A string that contains the next persistent variable name.
    //----------------------------------------------------------------------
    ConstString
    GetNextPersistentVariableName ();
    
    void
    RemovePersistentVariable (lldb::ClangExpressionVariableSP variable);

    void
    RegisterPersistentType (const ConstString &name,
                            clang::TypeDecl *tag_decl);
    
    clang::TypeDecl *
    GetPersistentType (const ConstString &name);
    
private:
    uint32_t                                                m_next_persistent_variable_id;  ///< The counter used by GetNextResultName().
    
    typedef llvm::DenseMap<const char *, clang::TypeDecl *> PersistentTypeMap;
    PersistentTypeMap                                       m_persistent_types;             ///< The persistent types declared by the user.
};

}

#endif
