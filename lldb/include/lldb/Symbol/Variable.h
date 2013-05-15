//===-- Variable.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Variable_h_
#define liblldb_Variable_h_

#include <vector>

#include "lldb/lldb-private.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/Core/Mangled.h"
#include "lldb/Core/UserID.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Symbol/Declaration.h"

namespace lldb_private {

class Variable : public UserID
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    Variable (lldb::user_id_t uid,
              const char *name, 
              const char *mangled,   // The mangled variable name for variables in namespaces
              const lldb::SymbolFileTypeSP &symfile_type_sp,
              lldb::ValueType scope,
              SymbolContextScope *owner_scope,
              Declaration* decl,
              const DWARFExpression& location,
              bool external,
              bool artificial);

    virtual
    ~Variable();

    void
    Dump(Stream *s, bool show_context) const;

    bool
    DumpDeclaration (Stream *s, 
                     bool show_fullpaths, 
                     bool show_module);
    
    const Declaration&
    GetDeclaration() const
    {
        return m_declaration;
    }

    const ConstString&
    GetName() const;

    SymbolContextScope *
    GetSymbolContextScope() const
    {
        return m_owner_scope;
    }

    // Since a variable can have a basename "i" and also a mangled 
    // named "_ZN12_GLOBAL__N_11iE" and a demangled mangled name 
    // "(anonymous namespace)::i", this function will allow a generic match
    // function that can be called by commands and expression parsers to make
    // sure we match anything we come across.
    bool
    NameMatches (const ConstString &name) const
    {
        if (m_name == name)
            return true;
        return m_mangled.NameMatches (name);
    }

    bool
    NameMatches (const RegularExpression& regex) const;

    Type *
    GetType();

    lldb::ValueType
    GetScope() const
    {
        return m_scope;
    }

    bool
    IsExternal() const
    {
        return m_external;
    }

    bool
    IsArtificial() const
    {
        return m_artificial;
    }

    DWARFExpression &
    LocationExpression()
    {
        return m_location;
    }

    const DWARFExpression &
    LocationExpression() const
    {
        return m_location;
    }
    
    bool
    DumpLocationForAddress (Stream *s, 
                            const Address &address);

    size_t
    MemorySize() const;

    void
    CalculateSymbolContext (SymbolContext *sc);

    bool
    IsInScope (StackFrame *frame);

    bool
    LocationIsValidForFrame (StackFrame *frame);

    bool
    LocationIsValidForAddress (const Address &address);
    
    bool
    GetLocationIsConstantValueData () const
    {
        return m_loc_is_const_data;
    }
    
    void
    SetLocationIsConstantValueData (bool b)
    {
        m_loc_is_const_data = b;
    }
    
    typedef size_t (*GetVariableCallback) (void *baton,
                                           const char *name,
                                           VariableList &var_list);


    static Error
    GetValuesForVariableExpressionPath (const char *variable_expr_path,
                                        ExecutionContextScope *scope,
                                        GetVariableCallback callback,
                                        void *baton,
                                        VariableList &variable_list,
                                        ValueObjectList &valobj_list);

    static size_t
    AutoComplete (const ExecutionContext &exe_ctx,
                  const char *name,
                  StringList &matches,
                  bool &word_complete);

protected:
    ConstString m_name;                 // The basename of the variable (no namespaces)
    Mangled m_mangled;                  // The mangled name of the variable
    lldb::SymbolFileTypeSP m_symfile_type_sp;   // The type pointer of the variable (int, struct, class, etc)
    lldb::ValueType m_scope;            // global, parameter, local
    SymbolContextScope *m_owner_scope;  // The symbol file scope that this variable was defined in
    Declaration m_declaration;          // Declaration location for this item.
    DWARFExpression m_location;         // The location of this variable that can be fed to DWARFExpression::Evaluate()
    uint8_t m_external:1,               // Visible outside the containing compile unit?
            m_artificial:1,             // Non-zero if the variable is not explicitly declared in source
            m_loc_is_const_data:1;      // The m_location expression contains the constant variable value data, not a DWARF location
private:
    Variable(const Variable& rhs);
    Variable& operator=(const Variable& rhs);
};

} // namespace lldb_private

#endif  // liblldb_Variable_h_
