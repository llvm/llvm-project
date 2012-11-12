//===-- VariableList.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_VariableList_h_
#define liblldb_VariableList_h_

#include "lldb/lldb-private.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Variable.h"

namespace lldb_private {

class VariableList
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
//  VariableList(const SymbolContext &symbol_context);
    VariableList();
    virtual ~VariableList();

    void
    AddVariable (const lldb::VariableSP &var_sp);

    bool
    AddVariableIfUnique (const lldb::VariableSP &var_sp);

    void
    AddVariables (VariableList *variable_list);

    void
    Clear();

    void
    Dump(Stream *s, bool show_context) const;

    lldb::VariableSP
    GetVariableAtIndex(uint32_t idx);

    lldb::VariableSP
    RemoveVariableAtIndex (uint32_t idx);
    
    lldb::VariableSP
    FindVariable (const ConstString& name);

    uint32_t
    FindVariableIndex (const lldb::VariableSP &var_sp);

    // Returns the actual number of unique variables that were added to the
    // list. "total_matches" will get updated with the actualy number of
    // matches that were found regardless of whether they were unique or not
    // to allow for error conditions when nothing is found, versus conditions
    // where any varaibles that match "regex" were already in "var_list".
    size_t
    AppendVariablesIfUnique (const RegularExpression& regex, 
                             VariableList &var_list, 
                             size_t& total_matches);

    uint32_t
    FindIndexForVariable (Variable* variable);

    size_t
    MemorySize() const;

    size_t
    GetSize() const;

protected:
    typedef std::vector<lldb::VariableSP> collection;
    typedef collection::iterator iterator;
    typedef collection::const_iterator const_iterator;

    collection m_variables;
private:
    //------------------------------------------------------------------
    // For VariableList only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (VariableList);
};

} // namespace lldb_private

#endif  // liblldb_VariableList_h_
