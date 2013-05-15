//===-- OptionValueProperties.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionValueProperties_h_
#define liblldb_OptionValueProperties_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ConstString.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Interpreter/OptionValue.h"
#include "lldb/Interpreter/Property.h"

namespace lldb_private {

class OptionValueProperties :
    public OptionValue,
    public std::enable_shared_from_this<OptionValueProperties>
{
public:

    //---------------------------------------------------------------------
    // OptionValueProperties
    //---------------------------------------------------------------------
    OptionValueProperties () :
        OptionValue(),
        m_name (),
        m_properties (),
        m_name_to_index ()
    {
    }

    OptionValueProperties (const ConstString &name);

    OptionValueProperties (const OptionValueProperties &global_properties);
    
    virtual
    ~OptionValueProperties()
    {
    }
    
    virtual Type
    GetType () const
    {
        return eTypeProperties;
    }
    
    virtual bool
    Clear ();

    virtual lldb::OptionValueSP
    DeepCopy () const;
    
    virtual Error
    SetValueFromCString (const char *value, VarSetOperationType op = eVarSetOperationAssign);

    virtual void
    DumpValue (const ExecutionContext *exe_ctx,
               Stream &strm,
               uint32_t dump_mask);

    virtual ConstString
    GetName () const
    {
        return m_name;
    }
    
    virtual Error
    DumpPropertyValue (const ExecutionContext *exe_ctx,
                       Stream &strm,
                       const char *property_path,
                       uint32_t dump_mask);

    virtual void
    DumpAllDescriptions (CommandInterpreter &interpreter,
                         Stream &strm) const;

    void
    Apropos (const char *keyword,
             std::vector<const Property *> &matching_properties) const;
    
    void
    Initialize (const PropertyDefinition *setting_definitions);

//    bool
//    GetQualifiedName (Stream &strm);

    //---------------------------------------------------------------------
    // Subclass specific functions
    //---------------------------------------------------------------------
    
    virtual size_t
    GetNumProperties() const;
    
    virtual ConstString
    GetPropertyNameAtIndex (uint32_t idx) const;

    virtual const char *
    GetPropertyDescriptionAtIndex (uint32_t idx) const;
    
    //---------------------------------------------------------------------
    // Get the index of a property given its exact name in this property
    // collection, "name" can't be a path to a property path that refers
    // to a property within a property
    //---------------------------------------------------------------------
    virtual uint32_t
    GetPropertyIndex (const ConstString &name) const;

    //---------------------------------------------------------------------
    // Get a property by exact name exists in this property collection, name
    // can not be a path to a property path that refers to a property within
    // a property
    //---------------------------------------------------------------------
    virtual const Property *
    GetProperty (const ExecutionContext *exe_ctx,
                 bool will_modify,
                 const ConstString &name) const;
    
    virtual const Property *
    GetPropertyAtIndex (const ExecutionContext *exe_ctx,
                        bool will_modify,
                        uint32_t idx) const;

    //---------------------------------------------------------------------
    // Property can be be a property path like "target.process.extra-startup-command"
    //---------------------------------------------------------------------
    virtual const Property *
    GetPropertyAtPath (const ExecutionContext *exe_ctx,
                       bool will_modify,
                       const char *property_path) const;

    virtual lldb::OptionValueSP
    GetPropertyValueAtIndex (const ExecutionContext *exe_ctx,
                             bool will_modify,
                             uint32_t idx) const;
    
    virtual lldb::OptionValueSP
    GetValueForKey  (const ExecutionContext *exe_ctx,
                     const ConstString &key,
                     bool value_will_be_modified) const;

    lldb::OptionValueSP
    GetSubValue (const ExecutionContext *exe_ctx,
                 const char *name,
                 bool value_will_be_modified,
                 Error &error) const;

    virtual Error
    SetSubValue (const ExecutionContext *exe_ctx,
                 VarSetOperationType op,
                 const char *path,
                 const char *value);

    virtual bool
    PredicateMatches (const ExecutionContext *exe_ctx,
                      const char *predicate) const
    {
        return false;
    }


    OptionValueArch *
    GetPropertyAtIndexAsOptionValueArch (const ExecutionContext *exe_ctx, uint32_t idx) const;

    bool
    GetPropertyAtIndexAsArgs (const ExecutionContext *exe_ctx, uint32_t idx, Args &args) const;

    bool
    SetPropertyAtIndexFromArgs (const ExecutionContext *exe_ctx, uint32_t idx, const Args &args);
    
    bool
    GetPropertyAtIndexAsBoolean (const ExecutionContext *exe_ctx, uint32_t idx, bool fail_value) const;
    
    bool
    SetPropertyAtIndexAsBoolean (const ExecutionContext *exe_ctx, uint32_t idx, bool new_value);
    
    OptionValueDictionary *
    GetPropertyAtIndexAsOptionValueDictionary (const ExecutionContext *exe_ctx, uint32_t idx) const;

    int64_t
    GetPropertyAtIndexAsEnumeration (const ExecutionContext *exe_ctx, uint32_t idx, int64_t fail_value) const;
    
    bool
    SetPropertyAtIndexAsEnumeration (const ExecutionContext *exe_ctx, uint32_t idx, int64_t new_value);
    
    const RegularExpression *
    GetPropertyAtIndexAsOptionValueRegex (const ExecutionContext *exe_ctx, uint32_t idx) const;

    OptionValueSInt64 *
    GetPropertyAtIndexAsOptionValueSInt64 (const ExecutionContext *exe_ctx, uint32_t idx) const;

    int64_t
    GetPropertyAtIndexAsSInt64 (const ExecutionContext *exe_ctx, uint32_t idx, int64_t fail_value) const;

    bool
    SetPropertyAtIndexAsSInt64 (const ExecutionContext *exe_ctx, uint32_t idx, int64_t new_value);

    uint64_t
    GetPropertyAtIndexAsUInt64 (const ExecutionContext *exe_ctx, uint32_t idx, uint64_t fail_value) const;
    
    bool
    SetPropertyAtIndexAsUInt64 (const ExecutionContext *exe_ctx, uint32_t idx, uint64_t new_value);
    
    const char *
    GetPropertyAtIndexAsString (const ExecutionContext *exe_ctx, uint32_t idx, const char *fail_value) const;

    bool
    SetPropertyAtIndexAsString (const ExecutionContext *exe_ctx, uint32_t idx, const char *new_value);

    OptionValueString *
    GetPropertyAtIndexAsOptionValueString (const ExecutionContext *exe_ctx, bool will_modify, uint32_t idx) const;

    OptionValueFileSpec *
    GetPropertyAtIndexAsOptionValueFileSpec (const ExecutionContext *exe_ctx, bool will_modify, uint32_t idx) const;

    FileSpec
    GetPropertyAtIndexAsFileSpec (const ExecutionContext *exe_ctx, uint32_t idx) const;
    
    bool
    SetPropertyAtIndexAsFileSpec (const ExecutionContext *exe_ctx, uint32_t idx, const FileSpec &file_spec);

    OptionValuePathMappings *
    GetPropertyAtIndexAsOptionValuePathMappings (const ExecutionContext *exe_ctx, bool will_modify, uint32_t idx) const;

    OptionValueFileSpecList *
    GetPropertyAtIndexAsOptionValueFileSpecList (const ExecutionContext *exe_ctx, bool will_modify, uint32_t idx) const;

    void
    AppendProperty(const ConstString &name,
                   const ConstString &desc,
                   bool is_global,
                   const lldb::OptionValueSP &value_sp);

    lldb::OptionValuePropertiesSP
    GetSubProperty (const ExecutionContext *exe_ctx,
                    const ConstString &name);

protected:
    
    const Property *
    ProtectedGetPropertyAtIndex (uint32_t idx) const
    {
        if (idx < m_properties.size())
            return &m_properties[idx];
        return NULL;
    }

    typedef UniqueCStringMap<size_t> NameToIndex;

    ConstString m_name;
    std::vector<Property> m_properties;
    NameToIndex m_name_to_index;
};

} // namespace lldb_private

#endif  // liblldb_OptionValueProperties_h_
