//===-- ObjCLanguageRuntime.h ---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ObjCLanguageRuntime_h_
#define liblldb_ObjCLanguageRuntime_h_

// C Includes
// C++ Includes
#include <functional>
#include <map>
#include <unordered_set>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeVendor.h"
#include "lldb/Target/LanguageRuntime.h"

namespace lldb_private {
    
class ClangUtilityFunction;

class ObjCLanguageRuntime :
    public LanguageRuntime
{
public:
    class MethodName
    {
    public:
        enum Type
        {
            eTypeUnspecified,
            eTypeClassMethod,
            eTypeInstanceMethod
        };
        
        MethodName () :
            m_full(),
            m_class(),
            m_category(),
            m_selector(),
            m_type (eTypeUnspecified),
            m_category_is_valid (false)
        {
        }

        MethodName (const char *name, bool strict) :
            m_full(),
            m_class(),
            m_category(),
            m_selector(),
            m_type (eTypeUnspecified),
            m_category_is_valid (false)
        {
            SetName (name, strict);
        }

        void
        Clear();

        bool
        IsValid (bool strict) const
        {
            // If "strict" is true, the name must have everything specified including
            // the leading "+" or "-" on the method name
            if (strict && m_type == eTypeUnspecified)
                return false;
            // Other than that, m_full will only be filled in if the objective C
            // name is valid.
            return (bool)m_full;
        }
        
        bool
        HasCategory()
        {
            return (bool)GetCategory();
        }

        Type
        GetType () const
        {
            return m_type;
        }
        
        const ConstString &
        GetFullName () const
        {
            return m_full;
        }
        
        ConstString
        GetFullNameWithoutCategory (bool empty_if_no_category);

        bool
        SetName (const char *name, bool strict);

        const ConstString &
        GetClassName ();

        const ConstString &
        GetClassNameWithCategory ();

        const ConstString &
        GetCategory ();
        
        const ConstString &
        GetSelector ();

        // Get all possible names for a method. Examples:
        // If name is "+[NSString(my_additions) myStringWithCString:]"
        //  names[0] => "+[NSString(my_additions) myStringWithCString:]"
        //  names[1] => "+[NSString myStringWithCString:]"
        // If name is specified without the leading '+' or '-' like "[NSString(my_additions) myStringWithCString:]"
        //  names[0] => "+[NSString(my_additions) myStringWithCString:]"
        //  names[1] => "-[NSString(my_additions) myStringWithCString:]"
        //  names[2] => "+[NSString myStringWithCString:]"
        //  names[3] => "-[NSString myStringWithCString:]"
        size_t
        GetFullNames (std::vector<ConstString> &names, bool append);
    protected:
        ConstString m_full;     // Full name:   "+[NSString(my_additions) myStringWithCString:]"
        ConstString m_class;    // Class name:  "NSString"
        ConstString m_class_category; // Class with category: "NSString(my_additions)"
        ConstString m_category; // Category:    "my_additions"
        ConstString m_selector; // Selector:    "myStringWithCString:"
        Type m_type;
        bool m_category_is_valid;

    };
    typedef lldb::addr_t ObjCISA;
    
    class ClassDescriptor;
    typedef std::shared_ptr<ClassDescriptor> ClassDescriptorSP;
    
    // the information that we want to support retrieving from an ObjC class
    // this needs to be pure virtual since there are at least 2 different implementations
    // of the runtime, and more might come
    class ClassDescriptor
    {
    public:
        
        ClassDescriptor() :
            m_is_kvo (eLazyBoolCalculate),
            m_is_cf (eLazyBoolCalculate),
            m_type_wp ()
        {
        }

        virtual
        ~ClassDescriptor ()
        {
        }
        
        virtual ConstString
        GetClassName () = 0;
        
        virtual ClassDescriptorSP
        GetSuperclass () = 0;
        
        // virtual if any implementation has some other version-specific rules
        // but for the known v1/v2 this is all that needs to be done
        virtual bool
        IsKVO ()
        {
            if (m_is_kvo == eLazyBoolCalculate)
            {
                const char* class_name = GetClassName().AsCString();
                if (class_name && *class_name)
                    m_is_kvo = (LazyBool)(strstr(class_name,"NSKVONotifying_") == class_name);
            }
            return (m_is_kvo == eLazyBoolYes);
        }
        
        // virtual if any implementation has some other version-specific rules
        // but for the known v1/v2 this is all that needs to be done
        virtual bool
        IsCFType ()
        {
            if (m_is_cf == eLazyBoolCalculate)
            {
                const char* class_name = GetClassName().AsCString();
                if (class_name && *class_name)
                    m_is_cf = (LazyBool)(strcmp(class_name,"__NSCFType") == 0 ||
                                         strcmp(class_name,"NSCFType") == 0);
            }
            return (m_is_cf == eLazyBoolYes);
        }
        
        virtual bool
        IsValid () = 0;
        
        virtual bool
        GetTaggedPointerInfo (uint64_t* info_bits = NULL,
                              uint64_t* value_bits = NULL,
                              uint64_t* payload = NULL) = 0;
        
        virtual uint64_t
        GetInstanceSize () = 0;
        
        // use to implement version-specific additional constraints on pointers
        virtual bool
        CheckPointer (lldb::addr_t value,
                      uint32_t ptr_size) const
        {
            return true;
        }
        
        virtual ObjCISA
        GetISA () = 0;
        
        // This should return true iff the interface could be completed
        virtual bool
        Describe (std::function <void (ObjCISA)> const &superclass_func,
                  std::function <bool (const char*, const char*)> const &instance_method_func,
                  std::function <bool (const char*, const char*)> const &class_method_func,
                  std::function <bool (const char *, const char *, lldb::addr_t, uint64_t)> const &ivar_func)
        {
            return false;
        }
        
        lldb::TypeSP
        GetType ()
        {
            return m_type_wp.lock();
        }
        
        void
        SetType (const lldb::TypeSP &type_sp)
        {
            m_type_wp = type_sp;
        }
        
    protected:
        bool
        IsPointerValid (lldb::addr_t value,
                        uint32_t ptr_size,
                        bool allow_NULLs = false,
                        bool allow_tagged = false,
                        bool check_version_specific = false) const;
        
    private:
        LazyBool m_is_kvo;
        LazyBool m_is_cf;
        lldb::TypeWP m_type_wp;
    };
    
    virtual ClassDescriptorSP
    GetClassDescriptor (ValueObject& in_value);
    
    ClassDescriptorSP
    GetNonKVOClassDescriptor (ValueObject& in_value);

    virtual ClassDescriptorSP
    GetClassDescriptorFromClassName (const ConstString &class_name);

    virtual ClassDescriptorSP
    GetClassDescriptorFromISA (ObjCISA isa);

    ClassDescriptorSP
    GetNonKVOClassDescriptor (ObjCISA isa);
    
    virtual
    ~ObjCLanguageRuntime();
    
    virtual lldb::LanguageType
    GetLanguageType () const
    {
        return lldb::eLanguageTypeObjC;
    }
    
    virtual bool
    IsModuleObjCLibrary (const lldb::ModuleSP &module_sp) = 0;
    
    virtual bool
    ReadObjCLibrary (const lldb::ModuleSP &module_sp) = 0;
    
    virtual bool
    HasReadObjCLibrary () = 0;
    
    virtual lldb::ThreadPlanSP
    GetStepThroughTrampolinePlan (Thread &thread, bool stop_others) = 0;

    lldb::addr_t
    LookupInMethodCache (lldb::addr_t class_addr, lldb::addr_t sel);

    void
    AddToMethodCache (lldb::addr_t class_addr, lldb::addr_t sel, lldb::addr_t impl_addr);
    
    TypeAndOrName
    LookupInClassNameCache (lldb::addr_t class_addr);
    
    void
    AddToClassNameCache (lldb::addr_t class_addr, const char *name, lldb::TypeSP type_sp);
    
    void
    AddToClassNameCache (lldb::addr_t class_addr, const TypeAndOrName &class_or_type_name);
    
    lldb::TypeSP
    LookupInCompleteClassCache (ConstString &name);
    
    virtual ClangUtilityFunction *
    CreateObjectChecker (const char *) = 0;
    
    virtual ObjCRuntimeVersions
    GetRuntimeVersion ()
    {
        return eObjC_VersionUnknown;
    }
        
    bool
    IsValidISA(ObjCISA isa)
    {
        UpdateISAToDescriptorMap();
        return m_isa_to_descriptor.count(isa) > 0;
    }

    virtual void
    UpdateISAToDescriptorMapIfNeeded() = 0;

    void
    UpdateISAToDescriptorMap()
    {
        if (m_process && m_process->GetStopID() != m_isa_to_descriptor_stop_id)
        {
            UpdateISAToDescriptorMapIfNeeded ();
        }
    }
    
    virtual ObjCISA
    GetISA(const ConstString &name);
    
    virtual ConstString
    GetActualTypeName(ObjCISA isa);
    
    virtual ObjCISA
    GetParentClass(ObjCISA isa);
    
    virtual TypeVendor *
    GetTypeVendor()
    {
        return NULL;
    }
    
    // Finds the byte offset of the child_type ivar in parent_type.  If it can't find the
    // offset, returns LLDB_INVALID_IVAR_OFFSET.
    
    virtual size_t
    GetByteOffsetForIvar (ClangASTType &parent_qual_type, const char *ivar_name);
    
    // Given the name of an Objective-C runtime symbol (e.g., ivar offset symbol),
    // try to determine from the runtime what the value of that symbol would be.
    // Useful when the underlying binary is stripped.
    virtual lldb::addr_t
    LookupRuntimeSymbol (const ConstString &name)
    {
        return LLDB_INVALID_ADDRESS;
    }
    
    //------------------------------------------------------------------
    /// Chop up an objective C function prototype.
    ///
    /// Chop up an objective C function fullname and optionally fill in
    /// any non-NULL ConstString objects. If a ConstString * is NULL,
    /// then this name doesn't get filled in
    ///
    /// @param[in] name
    ///     A fully specified objective C function name. The string might
    ///     contain a category and it includes the leading "+" or "-" and
    ///     the square brackets, no types for the arguments, just the plain
    ///     selector. A few examples:
    ///         "-[NSStringDrawingContext init]"
    ///         "-[NSStringDrawingContext addString:inRect:]"
    ///         "-[NSString(NSStringDrawing) sizeWithAttributes:]"
    ///         "+[NSString(NSStringDrawing) usesFontLeading]"
    ///         
    /// @param[out] class_name
    ///     If non-NULL, this string will be filled in with the class
    ///     name including the category. The examples above would return:
    ///         "NSStringDrawingContext"
    ///         "NSStringDrawingContext"
    ///         "NSString(NSStringDrawing)"
    ///         "NSString(NSStringDrawing)"
    ///
    /// @param[out] selector_name
    ///     If non-NULL, this string will be filled in with the selector
    ///     name. The examples above would return:
    ///         "init"
    ///         "addString:inRect:"
    ///         "sizeWithAttributes:"
    ///         "usesFontLeading"
    ///
    /// @param[out] name_sans_category
    ///     If non-NULL, this string will be filled in with the class
    ///     name _without_ the category. If there is no category, and empty
    ///     string will be returned (as the result would be normally returned
    ///     in the "class_name" argument). The examples above would return:
    ///         <empty>
    ///         <empty>
    ///         "-[NSString sizeWithAttributes:]"
    ///         "+[NSString usesFontLeading]"
    ///
    /// @param[out] class_name_sans_category
    ///     If non-NULL, this string will be filled in with the prototype
    ///     name _without_ the category. If there is no category, and empty
    ///     string will be returned (as this is already the value that was
    ///     passed in). The examples above would return:
    ///         <empty>
    ///         <empty>
    ///         "NSString"
    ///         "NSString"
    ///
    /// @return
    ///     Returns the number of strings that were successfully filled
    ///     in.
    //------------------------------------------------------------------
//    static uint32_t
//    ParseMethodName (const char *name, 
//                     ConstString *class_name,               // Class name (with category if there is one)
//                     ConstString *selector_name,            // selector only
//                     ConstString *name_sans_category,       // full function name with no category (empty if no category)
//                     ConstString *class_name_sans_category);// Class name without category (empty if no category)
    
    static bool
    IsPossibleObjCMethodName (const char *name)
    {
        if (!name)
            return false;
        bool starts_right = (name[0] == '+' || name[0] == '-') && name[1] == '[';
        bool ends_right = (name[strlen(name) - 1] == ']');
        return (starts_right && ends_right);
    }
    
    static bool
    IsPossibleObjCSelector (const char *name)
    {
        if (!name)
            return false;
            
        if (strchr(name, ':') == NULL)
            return true;
        else if (name[strlen(name) - 1] == ':')
            return true;
        else
            return false;
    }
    
    bool
    HasNewLiteralsAndIndexing ()
    {
        if (m_has_new_literals_and_indexing == eLazyBoolCalculate)
        {
            if (CalculateHasNewLiteralsAndIndexing())
                m_has_new_literals_and_indexing = eLazyBoolYes;
            else
                m_has_new_literals_and_indexing = eLazyBoolNo;
        }
        
        return (m_has_new_literals_and_indexing == eLazyBoolYes);
    }
    
    virtual void
    SymbolsDidLoad (const ModuleList& module_list)
    {
        m_negative_complete_class_cache.clear();
    }
    
protected:
    //------------------------------------------------------------------
    // Classes that inherit from ObjCLanguageRuntime can see and modify these
    //------------------------------------------------------------------
    ObjCLanguageRuntime(Process *process);
    
    virtual bool CalculateHasNewLiteralsAndIndexing()
    {
        return false;
    }
    
    
    bool
    ISAIsCached (ObjCISA isa) const
    {
        return m_isa_to_descriptor.find(isa) != m_isa_to_descriptor.end();
    }

    bool
    AddClass (ObjCISA isa, const ClassDescriptorSP &descriptor_sp)
    {
        if (isa != 0)
        {
            m_isa_to_descriptor[isa] = descriptor_sp;
            return true;
        }
        return false;
    }

    bool
    AddClass (ObjCISA isa, const ClassDescriptorSP &descriptor_sp, const char *class_name);

    bool
    AddClass (ObjCISA isa, const ClassDescriptorSP &descriptor_sp, uint32_t class_name_hash)
    {
        if (isa != 0)
        {
            m_isa_to_descriptor[isa] = descriptor_sp;
            m_hash_to_isa_map.insert(std::make_pair(class_name_hash, isa));
            return true;
        }
        return false;
    }

private:
    // We keep a map of <Class,Selector>->Implementation so we don't have to call the resolver
    // function over and over.
    
    // FIXME: We need to watch for the loading of Protocols, and flush the cache for any
    // class that we see so changed.
    
    struct ClassAndSel
    {
        ClassAndSel()
        {
            sel_addr = LLDB_INVALID_ADDRESS;
            class_addr = LLDB_INVALID_ADDRESS;
        }
        ClassAndSel (lldb::addr_t in_sel_addr, lldb::addr_t in_class_addr) :
            class_addr (in_class_addr),
            sel_addr(in_sel_addr)
        {
        }
        bool operator== (const ClassAndSel &rhs)
        {
            if (class_addr == rhs.class_addr
                && sel_addr == rhs.sel_addr)
                return true;
            else
                return false;
        }
        
        bool operator< (const ClassAndSel &rhs) const
        {
            if (class_addr < rhs.class_addr)
                return true;
            else if (class_addr > rhs.class_addr)
                return false;
            else
            {
                if (sel_addr < rhs.sel_addr)
                    return true;
                else
                    return false;
            }
        }
        
        lldb::addr_t class_addr;
        lldb::addr_t sel_addr;
    };

    typedef std::map<ClassAndSel,lldb::addr_t> MsgImplMap;
    typedef std::map<ObjCISA, ClassDescriptorSP> ISAToDescriptorMap;
    typedef std::multimap<uint32_t, ObjCISA> HashToISAMap;
    typedef ISAToDescriptorMap::iterator ISAToDescriptorIterator;
    typedef HashToISAMap::iterator HashToISAIterator;

    MsgImplMap m_impl_cache;
    LazyBool m_has_new_literals_and_indexing;
    ISAToDescriptorMap m_isa_to_descriptor;
    HashToISAMap m_hash_to_isa_map;

protected:
    uint32_t m_isa_to_descriptor_stop_id;

    typedef std::map<ConstString, lldb::TypeWP> CompleteClassMap;
    CompleteClassMap m_complete_class_cache;
    
    struct ConstStringSetHelpers {
        size_t operator () (const ConstString& arg) const // for hashing
        {
            return (size_t)arg.GetCString();
        }
        bool operator () (const ConstString& arg1, const ConstString& arg2) const // for equality
        {
            return arg1.operator==(arg2);
        }
    };
    typedef std::unordered_set<ConstString, ConstStringSetHelpers, ConstStringSetHelpers> CompleteClassSet;
    CompleteClassSet m_negative_complete_class_cache;

    ISAToDescriptorIterator
    GetDescriptorIterator (const ConstString &name);

    DISALLOW_COPY_AND_ASSIGN (ObjCLanguageRuntime);
};

} // namespace lldb_private

#endif  // liblldb_ObjCLanguageRuntime_h_
