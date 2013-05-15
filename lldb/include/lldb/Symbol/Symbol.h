//===-- Symbol.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Symbol_h_
#define liblldb_Symbol_h_

#include "lldb/lldb-private.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/Mangled.h"
#include "lldb/Core/UserID.h"
#include "lldb/Symbol/SymbolContextScope.h"

namespace lldb_private {

class Symbol :
    public SymbolContextScope
{
public:
    // ObjectFile readers can classify their symbol table entries and searches can be made
    // on specific types where the symbol values will have drastically different meanings
    // and sorting requirements.
    Symbol();

    Symbol (uint32_t symID,
            const char *name,
            bool name_is_mangled,
            lldb::SymbolType type,
            bool external,
            bool is_debug,
            bool is_trampoline,
            bool is_artificial,
            const lldb::SectionSP &section_sp,
            lldb::addr_t value,
            lldb::addr_t size,
            bool size_is_valid,
            uint32_t flags);

    Symbol (uint32_t symID,
            const char *name,
            bool name_is_mangled,
            lldb::SymbolType type,
            bool external,
            bool is_debug,
            bool is_trampoline,
            bool is_artificial,
            const AddressRange &range,
            bool size_is_valid,
            uint32_t flags);

    Symbol (const Symbol& rhs);

    const Symbol&
    operator= (const Symbol& rhs);

    void
    Clear();

    bool
    Compare (const ConstString& name, lldb::SymbolType type) const;

    void
    Dump (Stream *s, Target *target, uint32_t index) const;

    bool
    ValueIsAddress() const;

    //------------------------------------------------------------------
    // Access the address value. Do NOT hand out the AddressRange as an
    // object as the byte size of the address range may not be filled in
    // and it should be accessed via GetByteSize().
    //------------------------------------------------------------------
    Address &
    GetAddress()
    {
        return m_addr_range.GetBaseAddress();
    }

    //------------------------------------------------------------------
    // Access the address value. Do NOT hand out the AddressRange as an
    // object as the byte size of the address range may not be filled in
    // and it should be accessed via GetByteSize().
    //------------------------------------------------------------------
    const Address &
    GetAddress() const
    {
        return m_addr_range.GetBaseAddress();
    }

    const ConstString &
    GetName () const
    {
        return m_mangled.GetName();
    }

    uint32_t
    GetID() const
    {
        return m_uid;
    }

    void
    SetID(uint32_t uid)
    {
        m_uid = uid;
    }

    Mangled&
    GetMangled ()
    {
        return m_mangled;
    }

    const Mangled&
    GetMangled () const
    {
        return m_mangled;
    }

    uint32_t
    GetSiblingIndex () const;

    lldb::SymbolType
    GetType () const
    {
        return (lldb::SymbolType)m_type;
    }

    void
    SetType (lldb::SymbolType type)
    {
        m_type = (lldb::SymbolType)type;
    }

    const char *
    GetTypeAsString () const;

    uint32_t
    GetFlags () const
    {
        return m_flags;
    }

    void
    SetFlags (uint32_t flags)
    {
        m_flags = flags;
    }

    void
    GetDescription (Stream *s, lldb::DescriptionLevel level, Target *target) const;

    bool
    IsSynthetic () const
    {
        return m_is_synthetic;
    }

    void
    SetIsSynthetic (bool b)
    {
        m_is_synthetic = b;
    }

    
    bool
    GetSizeIsSynthesized() const
    {
        return m_size_is_synthesized;
    }
    
    void
    SetSizeIsSynthesized(bool b)
    {
        m_size_is_synthesized = b;
    }

    bool
    IsDebug () const
    {
        return m_is_debug;
    }

    void
    SetDebug (bool b)
    {
        m_is_debug = b;
    }

    bool
    IsExternal () const
    {
        return m_is_external;
    }

    void
    SetExternal (bool b)
    {
        m_is_external = b;
    }

    bool
    IsTrampoline () const;

    bool
    IsIndirect () const;

    lldb::addr_t
    GetByteSize () const;
    
    void
    SetByteSize (lldb::addr_t size)
    {
        m_calculated_size = size > 0;
        m_addr_range.SetByteSize(size);
    }

    bool
    GetSizeIsSibling () const
    {
        return m_size_is_sibling;
    }

    void
    SetSizeIsSibling (bool b)
    {
        m_size_is_sibling = b;
    }

//    void
//    SetValue (Address &value)
//    {
//        m_addr_range.GetBaseAddress() = value;
//    }
//
//    void
//    SetValue (const AddressRange &range)
//    {
//        m_addr_range = range;
//    }
//
//    void
//    SetValue (lldb::addr_t value);
//    {
//        m_addr_range.GetBaseAddress().SetRawAddress(value);
//    }

    // If m_type is "Code" or "Function" then this will return the prologue size
    // in bytes, else it will return zero.
    uint32_t
    GetPrologueByteSize ();

    bool
    GetDemangledNameIsSynthesized() const
    {
        return m_demangled_is_synthesized;
    }
    void
    SetDemangledNameIsSynthesized(bool b)
    {
        m_demangled_is_synthesized = b;
    }

    //------------------------------------------------------------------
    /// @copydoc SymbolContextScope::CalculateSymbolContext(SymbolContext*)
    ///
    /// @see SymbolContextScope
    //------------------------------------------------------------------
    virtual void
    CalculateSymbolContext (SymbolContext *sc);

    virtual lldb::ModuleSP
    CalculateSymbolContextModule ();
    
    virtual Symbol *
    CalculateSymbolContextSymbol ();

    //------------------------------------------------------------------
    /// @copydoc SymbolContextScope::DumpSymbolContext(Stream*)
    ///
    /// @see SymbolContextScope
    //------------------------------------------------------------------
    virtual void
    DumpSymbolContext (Stream *s);

protected:

    uint32_t        m_uid;                  // User ID (usually the original symbol table index)
    Mangled         m_mangled;              // uniqued symbol name/mangled name pair
    uint16_t        m_type_data;            // data specific to m_type
    uint16_t        m_type_data_resolved:1, // True if the data in m_type_data has already been calculated
                    m_is_synthetic:1,       // non-zero if this symbol is not actually in the symbol table, but synthesized from other info in the object file.
                    m_is_debug:1,           // non-zero if this symbol is debug information in a symbol
                    m_is_external:1,        // non-zero if this symbol is globally visible
                    m_size_is_sibling:1,    // m_size contains the index of this symbol's sibling
                    m_size_is_synthesized:1,// non-zero if this symbol's size was calculated using a delta between this symbol and the next
                    m_calculated_size:1,
                    m_demangled_is_synthesized:1, // The demangled name was created should not be used for expressions or other lookups
                    m_type:8;
    uint32_t        m_flags;                // A copy of the flags from the original symbol table, the ObjectFile plug-in can interpret these
    AddressRange    m_addr_range;           // Contains the value, or the section offset address when the value is an address in a section, and the size (if any)
};

} // namespace lldb_private

#endif  // liblldb_Symbol_h_
