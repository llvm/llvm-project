//===-- SwiftLanguageRuntime.h ----------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftLanguageRuntime_h_
#define liblldb_SwiftLanguageRuntime_h_

// C Includes
// C++ Includes
#include <vector>
// Other libraries and framework includes
// Project includes
#include "lldb/Core/PluginInterface.h"
#include "lldb/lldb-private.h"
#include "Plugins/LanguageRuntime/ObjC/AppleObjCRuntime/AppleObjCRuntimeV2.h"
#include "lldb/Target/LanguageRuntime.h"

#include "llvm/Support/Casting.h"
#include "llvm/ADT/Optional.h"

namespace lldb_private {
    
class SwiftLanguageRuntime :
    public LanguageRuntime
{
public:

    class Metadata;
    typedef std::shared_ptr<Metadata> MetadataSP;
    
    class GenericPattern;
    typedef std::shared_ptr<GenericPattern> GenericPatternSP;
    
    class NominalTypeDescriptor;
    typedef std::shared_ptr<NominalTypeDescriptor> NominalTypeDescriptorSP;
    
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();
    
    static void
    Terminate();
    
    static lldb_private::LanguageRuntime *
    CreateInstance (Process *process, lldb::LanguageType language);
    
    static lldb_private::ConstString
    GetPluginNameStatic();
    
    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    lldb_private::ConstString
    GetPluginName() override;
    
    uint32_t
    GetPluginVersion() override;
    
    class MethodName
    {
    public:
        enum Type
        {
            eTypeInvalid,
            eTypeUnknownMethod,
            eTypeClassMethod,
            eTypeInstanceMethod,
            eTypeOperator,
            eTypeConstructor,
            eTypeDestructor,
            eTypeAllocator,
            eTypeDeallocator
        };
        
        MethodName () :
        m_full(),
        m_basename(),
        m_context(),
        m_arguments(),
        m_qualifiers(),
        m_type (eTypeInvalid),
        m_parsed (false),
        m_parse_error (false)
        {
        }
        
        MethodName (const ConstString &s,
                    bool do_parse = false) :
        m_full(s),
        m_basename(),
        m_context(),
        m_arguments(),
        m_qualifiers(),
        m_type (eTypeInvalid),
        m_parsed (false),
        m_parse_error (false)
        {
            if (do_parse)
                Parse ();
        }
        
        void
        Clear();
        
        bool
        IsValid () const
        {
            if (m_parse_error)
                return false;
            if (m_type == eTypeInvalid)
                return false;
            return (bool)m_full;
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
        
        llvm::StringRef
        GetBasename ();
        
        llvm::StringRef
        GetContext ();
        
        llvm::StringRef
        GetMetatypeReference ();
        
        llvm::StringRef
        GetTemplateArguments ();
        
        llvm::StringRef
        GetArguments ();
        
        llvm::StringRef
        GetQualifiers ();
        
        llvm::StringRef
        GetReturnType ();

        static bool
        ExtractFunctionBasenameFromMangled (const ConstString &mangled,
                                            ConstString &basename,
                                            bool &is_method);
        
    protected:
        void
        Parse();
        
        ConstString     m_full;          // Full name:    "foo.bar.baz : <A : AProtocol> (foo.bar.metatype)(x : Swift.Int64) -> A"
        llvm::StringRef m_basename;      // Basename:     "baz"
        llvm::StringRef m_context;       // Decl context: "foo.bar"
        llvm::StringRef m_metatype_ref;  // Meta type:    "(foo.bar.metatype)"
        llvm::StringRef m_template_args; // Generic args: "<A: AProtocol>
        llvm::StringRef m_arguments;     // Arguments:    "(x : Swift.Int64)"
        llvm::StringRef m_qualifiers;    // Qualifiers:   "const"
        llvm::StringRef m_return_type;   // Return type:  "A"
        Type m_type;
        bool m_parsed;
        bool m_parse_error;
    };
    
    class MetadataUtils
    {
        friend class SwiftLanguageRuntime;
    public:
        MetadataUtils (SwiftLanguageRuntime& runtime,
                       lldb::addr_t base_addr);
        
        virtual ~MetadataUtils () = default;
        
        // this is an helper function that can be used to just read an integer
        // at some offset from the metadata pointer
        uint64_t
        ReadIntegerAtOffset (size_t size, int32_t offset);
        
        // this is an helper function that can be used to just read an integer
        // at some offset from the metadata pointer
        uint64_t
        ReadIntegerAtOffset (size_t size, lldb::addr_t base, int32_t offset);
        
        // this is an helper function that can be used to just read a pointer
        // at some offset from the metadata pointer
        // since most fields in Swift metadata are pointer sized and at some well-known
        // or easily computable number of pointer sized words from the base pointer,
        // offset is measured in pointer sized words instead of bytes
        lldb::addr_t
        ReadPointerAtOffset (int32_t offset);
        
        // this has the same rationale, but is useful to read at arbitrary locations
        // instead of basing off the metadata pointer - mostly useful if some metadata entry
        // ends up being a pointer-to interesting stuff worth reading
        lldb::addr_t
        ReadPointerAtOffset (lldb::addr_t base, int32_t offset);
        
        // this is an helper function to read a string whose pointer is at some offset
        bool
        ReadStringPointedAtOffset (int32_t offset, std::string& out_str);
        
        // this is an helper function to read a set of consecutive strings terminated by a double NULL
        // this is a fairly frequent metadata construct in Swift, so let's make it easy to parse it
        std::vector<std::string>
        ReadDoublyTerminatedStringList (lldb::addr_t location);
        
    protected:
        lldb::addr_t m_base_address;
        SwiftLanguageRuntime& m_runtime;
    };
    
    class NominalTypeDescriptor : protected MetadataUtils
    {
    friend class SwiftLanguageRuntime;
    public:
        NominalTypeDescriptor (SwiftLanguageRuntime& runtime,
                               lldb::addr_t base_addr);
        
        virtual ~NominalTypeDescriptor () = default;
        
        enum class Kind : uint64_t {
            Class = 0,
            Struct = 1,
            Enum = 2,
            Unknown = 0xFF
        };
        
        Kind
        GetKind () const
        {
            return m_kind;
        }
        
        std::string
        GetMangledName () const
        {
            return m_mangled_name;
            
        }
        
        lldb::addr_t
        GetGenericParameterVectorOffset () const
        {
            return m_gpv_offset;
        }
        
        size_t
        GetNumTypeParameters () const
        {
            return m_num_type_params;
        }
        
        lldb::addr_t
        GetNumWitnessesAtIndex (size_t idx) const
        {
            if (idx >= m_num_type_params)
                return LLDB_INVALID_ADDRESS;
            return m_num_witnesses[idx];
        }
        
        lldb::addr_t
        GetBaseAddress () const
        {
            return m_base_address;
        }
        
        static bool classof(const NominalTypeDescriptor *S)
        {
            return S->GetKind() == Kind::Unknown;
        }
        
    protected:
        Kind m_kind;
        std::string m_mangled_name;
        lldb::addr_t m_gpv_offset;
        size_t m_num_type_params;
        std::vector<size_t> m_num_witnesses;
        
    private:
        DISALLOW_COPY_AND_ASSIGN (NominalTypeDescriptor);
        
        virtual bool
        IsValid ()
        {
            return true;
        }
    };
    
    class AggregateNominalTypeDescriptor : public NominalTypeDescriptor
    {
    public:
        AggregateNominalTypeDescriptor (SwiftLanguageRuntime& runtime,
                                        lldb::addr_t base_addr);
        
        virtual ~AggregateNominalTypeDescriptor () = default;

        static bool classof(const NominalTypeDescriptor *S)
        {
            return S->GetKind() == Kind::Class ||
            S->GetKind() == Kind::Struct;
        }
        
        size_t
        GetNumFields () const
        {
            return m_num_fields;
        }
        
        lldb::addr_t
        GetFieldOffsetVectorOffset () const
        {
            return m_field_off_vec_offset;
        }
        
        std::string
        GetFieldNameAtIndex (size_t idx) const
        {
            if (idx >= m_num_fields)
                return "";
            return m_field_names[idx];
        }

    protected:
        size_t m_num_fields;
        lldb::addr_t m_field_off_vec_offset;
        std::vector<std::string> m_field_names;
        lldb::addr_t m_field_metadata_generator;
    private:
        DISALLOW_COPY_AND_ASSIGN(AggregateNominalTypeDescriptor);
        
        virtual bool
        IsValid ()
        {
            return m_num_fields == m_field_names.size();
        }
    };
    
    class EnumNominalTypeDescriptor : public NominalTypeDescriptor
    {
    public:
        EnumNominalTypeDescriptor (SwiftLanguageRuntime& runtime,
                                   lldb::addr_t base_addr);
        
        virtual ~EnumNominalTypeDescriptor () = default;

        static bool classof(const NominalTypeDescriptor *S)
        {
            return S->GetKind() == Kind::Enum;
        }
        
        size_t
        GetNumCases (bool nonempty = true, bool empty = true) const
        {
            return (empty ? m_num_empty_cases : 0) + (nonempty ? m_num_nonempty_cases : 0);
        }
        
        std::string
        GetCaseNameAtIndex (size_t idx) const
        {
            if (idx >= m_num_nonempty_cases + m_num_empty_cases)
                return "";
            return m_case_names[idx];
        }

    protected:
        size_t m_num_nonempty_cases;
        size_t m_num_empty_cases;
        std::vector<std::string> m_case_names;
        lldb::addr_t m_case_metadata_generator;

    private:
        DISALLOW_COPY_AND_ASSIGN(EnumNominalTypeDescriptor);
        
        virtual bool
        IsValid ()
        {
            return m_case_names.size() == GetNumCases(true,true);
        }
    };
    
    // named GenericMetadata in swiftlang
    class GenericPattern : protected MetadataUtils
    {
    friend class SwiftLanguageRuntime;
    public:
        GenericPattern (SwiftLanguageRuntime& runtime,
                        lldb::addr_t base_addr);
        
        virtual ~GenericPattern () = default;
        
        MetadataSP
        FindGenericMetadata(CompilerType type);
        
        lldb::addr_t
        GetFillFunctionAddress ()
        {
            return m_fill_function_addr;
        }
        
        uint32_t
        GetSize ()
        {
            return m_size;
        }
        
        uint16_t
        GetNumKeyArguments ()
        {
            return m_num_key_args;
        }
        
        lldb::addr_t
        GetCacheAddress ();
        
    protected:
        MetadataSP
        FindGenericMetadata(std::vector<CompilerType> args);

        class CacheAddressStore
        {
        public:
            CacheAddressStore();
            
            lldb::addr_t
            GetCacheAddress(const Process& p, std::function<lldb::addr_t(void)> f);
            
        private:
            lldb::addr_t m_cache_addr;
            uint32_t m_last_cache_update_stop_id;
        };
        
        lldb::addr_t m_fill_function_addr;
        uint32_t m_size;
        uint16_t m_num_key_args;
        CacheAddressStore m_cache_addr_store;
    };

    class Metadata : protected MetadataUtils
    {
    friend class SwiftLanguageRuntime;
    public:
        enum class Kind {
            Unknown = 0,
            Struct = 1,
            Enum = 2,
            Opaque = 8,
            Tuple = 9,
            Function = 10,
            Protocol = 12,
            Metatype = 13,
            ObjCWrapper = 14,
            Class = 4095
        };
        
        static bool classof(const Metadata *S)
        {
            return S->GetKind() == Kind::Unknown;
        }
        
        Kind
        GetKind () const
        {
            return m_kind;
        }
        
        lldb::addr_t
        GetBaseAddress () const
        {
            return m_base_address;
        }
        
        // TODO: if we eventually care about the VWT, return a struct
        // that defines it instead of just the pointer-to
        lldb::addr_t
        GetValueWitnessTableAddress () const
        {
            return m_value_witness_table;
        }
        
        // does this thing support being "sub-typed"?
        virtual bool
        IsStaticallyDetermined ()
        {
            return true;
        }
        
    protected:
        
        Metadata (SwiftLanguageRuntime& runtime,
                  lldb::addr_t base_addr);
        
    private:
        // TODO: do we want to read a value_witness?
        lldb::addr_t m_value_witness_table;
        Kind m_kind;
        
        DISALLOW_COPY_AND_ASSIGN(Metadata);
    };
    
    class GenericParameterVector : protected MetadataUtils
    {
    friend class SwiftLanguageRuntime;
    public:
        class GenericParameter
        {
        private:
            MetadataSP m_metadata_sp;
            
        public:
            GenericParameter (MetadataSP metadata) :
            m_metadata_sp(metadata)
            { }
            
            GenericParameter () :
            m_metadata_sp ()
            { }
            
            bool
            IsValid ();
            
            MetadataSP
            GetMetadata ()
            {
                return m_metadata_sp;
            }
        };
        
        size_t
        GetNumParameters ()
        {
            return m_num_primary_params;
        }
        
        GenericParameter
        GetParameterAtIndex (size_t);
        
    protected:
        GenericParameterVector (SwiftLanguageRuntime& runtime,
                                lldb::addr_t nom_type_desc_addr,
                                lldb::addr_t base_addr);
    private:
        size_t m_num_primary_params;
        std::vector<GenericParameter> m_params;
        
        DISALLOW_COPY_AND_ASSIGN(GenericParameterVector);
    };
    
    class NominalTypeMetadata : public Metadata
    {
    friend class SwiftLanguageRuntime;
    public:
        static bool classof(const Metadata *S)
        {
            auto kind = S->GetKind();
            switch (kind)
            {
                case Kind::Struct:
                case Kind::Class:
                case Kind::Enum:
                case Kind::ObjCWrapper:
                    return true;
                default:
                    return false;
            }
        }
        
        virtual std::string
        GetMangledName () = 0;
        
        virtual GenericParameterVector*
        GetGenericParameterVector () = 0;

    protected:
        NominalTypeMetadata (SwiftLanguageRuntime& runtime,
                             lldb::addr_t base_addr) :
        Metadata (runtime,base_addr) {}

    private:
        
        DISALLOW_COPY_AND_ASSIGN(NominalTypeMetadata);
    };
    
    class FieldContainerTypeMetadata : public NominalTypeMetadata
    {
    friend class SwiftLanguageRuntime;
    public:
        static bool classof(const Metadata *S)
        {
            auto kind = S->GetKind();
            switch (kind)
            {
                case Kind::Struct:
                case Kind::Class:
                    return true;
                default:
                    return false;
            }
        }
        
        class Field
        {
        private:
            ConstString m_name;
            lldb::addr_t m_offset;
            
        public:
            Field (std::string name, lldb::addr_t offset) :
            m_name(name.c_str()),
            m_offset(offset)
            { }
            
            Field () :
            m_name(),
            m_offset (LLDB_INVALID_ADDRESS)
            { }
            
            bool
            IsValid ();
            
            ConstString
            GetName ()
            {
                return m_name;
            }
            
            lldb::addr_t
            GetOffset ()
            {
                return m_offset;
            }
        };
        
        virtual size_t
        GetNumFields () = 0;
        
        virtual Field
        GetFieldAtIndex (size_t) = 0;

    protected:
        FieldContainerTypeMetadata (SwiftLanguageRuntime& runtime,
                                    lldb::addr_t base_addr) :
        NominalTypeMetadata (runtime,base_addr) {}
        
    private:
        DISALLOW_COPY_AND_ASSIGN(FieldContainerTypeMetadata);
    };
    
    class StructMetadata : public FieldContainerTypeMetadata
    {
    friend class SwiftLanguageRuntime;
    public:
        static bool classof(const Metadata *S)
        {
            return S->GetKind() == Kind::Struct;
        }
        
        virtual std::string
        GetMangledName ()
        {
            return m_mangled_name;
        }
        
        size_t
        GetNumFields ()
        {
            return m_num_fields;
        }
        
        Field
        GetFieldAtIndex (size_t);
        
        virtual GenericParameterVector*
        GetGenericParameterVector ()
        {
            return m_gpv_ap.get();
        }
        
    protected:
        StructMetadata (SwiftLanguageRuntime& runtime,
                        lldb::addr_t base_addr);

    private:
        std::string m_mangled_name;
        size_t m_num_fields;
        std::vector<Field> m_fields;
        std::unique_ptr<GenericParameterVector> m_gpv_ap;

        DISALLOW_COPY_AND_ASSIGN(StructMetadata);
    };
    
    class EnumMetadata : public NominalTypeMetadata
    {
    friend class SwiftLanguageRuntime;
    public:
        static bool classof(const Metadata *S)
        {
            return S->GetKind() == Kind::Enum;
        }
        
        virtual std::string
        GetMangledName ()
        {
            return m_mangled_name;
        }
        
        size_t
        GetNumCases ()
        {
            return m_num_nonempty_cases+m_num_empty_cases;
        }
        
        size_t
        GetNumNonEmptyCases ()
        {
            return m_num_nonempty_cases;
        }
        
        size_t
        GetNumEmptyCases ()
        {
            return m_num_empty_cases;
        }
        
        class Case
        {
        private:
            std::string m_name;
            bool m_empty;
            
        public:
            Case (std::string name, bool empty) :
            m_name(name),
            m_empty(empty)
            { }
            
            Case () :
            m_name(),
            m_empty()
            { }
            
            bool
            IsValid ();
            
            std::string
            GetName ()
            {
                return m_name;
            }
            
            bool
            IsEmpty ()
            {
                return m_empty;
            }
        };
        
        Case
        GetCaseAtIndex (size_t);
        
        virtual GenericParameterVector*
        GetGenericParameterVector ()
        {
            return m_gpv_ap.get();
        }
        
    protected:
        EnumMetadata (SwiftLanguageRuntime& runtime,
                      lldb::addr_t base_addr);
        
    private:
        std::string m_mangled_name;
        size_t m_num_nonempty_cases;
        size_t m_num_empty_cases;
        std::vector<Case> m_cases;
        std::unique_ptr<GenericParameterVector> m_gpv_ap;
        
        DISALLOW_COPY_AND_ASSIGN(EnumMetadata);
    };

    class OpaqueMetadata : public Metadata
    {
        friend class SwiftLanguageRuntime;
    public:
        static bool classof(const Metadata *S)
        {
            return S->GetKind() == Kind::Opaque;
        }
        
    protected:
        OpaqueMetadata (SwiftLanguageRuntime& runtime,
                       lldb::addr_t base_addr);
        
    private:
        DISALLOW_COPY_AND_ASSIGN(OpaqueMetadata);
    };
    
    class TupleMetadata : public Metadata
    {
        friend class SwiftLanguageRuntime;
    public:
        class Element
        {
        private:
            MetadataSP m_metadata_sp;
            std::string m_name;
            lldb::addr_t m_offset;
            
        public:
            Element (MetadataSP metadata,
                     std::string name,
                     lldb::addr_t offset) :
            m_metadata_sp (metadata),
            m_name(name),
            m_offset (offset)
            { }
            
            Element () :
            m_metadata_sp (),
            m_name(),
            m_offset (LLDB_INVALID_ADDRESS)
            { }
            
            bool
            IsValid ();
            
            MetadataSP
            GetMetadata () const
            {
                return m_metadata_sp;
            }
            
            const char*
            GetName () const
            {
                return m_name.c_str();
            }
            
            lldb::addr_t
            GetOffset () const
            {
                return m_offset;
            }
        };
        
        static bool classof(const Metadata *S)
        {
            return S->GetKind() == Kind::Tuple;
        }
        
        size_t
        GetNumElements ()
        {
            return m_num_elements;
        }
        
        Element
        GetElementAtIndex (size_t i);
        
    protected:
        
        TupleMetadata (SwiftLanguageRuntime& runtime,
                       lldb::addr_t base_addr);
        
    private:
        size_t m_num_elements;
        std::vector<Element> m_elements;
        
        DISALLOW_COPY_AND_ASSIGN(TupleMetadata);
    };
    
    class FunctionMetadata : public Metadata
    {
    friend class SwiftLanguageRuntime;
    public:
        static bool classof(const Metadata *S)
        {
            return S->GetKind() == Kind::Function;
        }
        
        MetadataSP
        GetArgumentMetadata ();
        
        MetadataSP
        GetReturnMetadata ();
        
        bool
        IsThrowsFunction ();
        
    protected:
        
        FunctionMetadata (SwiftLanguageRuntime& runtime,
                          lldb::addr_t base_addr);
        
    private:
        union {
            struct {
                uint32_t m_argc: 31;
                bool m_throws : 1;
            };
            uint32_t m_argc_and_throws;
        } m_argc_and_throws;
        MetadataSP m_arg_metadata_sp;
        MetadataSP m_ret_metadata_sp;
        
        DISALLOW_COPY_AND_ASSIGN(FunctionMetadata);
    };
    
    class ProtocolMetadata : public Metadata
    {
    friend class SwiftLanguageRuntime;
    public:
        static bool classof(const Metadata *S)
        {
            return S->GetKind() == Kind::Protocol;
        }
        
        // this is-not a metadata, but we like to have
        // the Read* utils - FIXME better design here?
        // like a MetadataUtils?
        class Descriptor : protected MetadataUtils
        {
        private:
            lldb::addr_t m_isa_placeholder;
            std::string m_mangled_name;
            std::vector<Descriptor> m_parents;
            lldb::addr_t m_required_instance_methods_addr;
            lldb::addr_t m_required_class_methods_addr;
            lldb::addr_t m_optional_instance_methods_addr;
            lldb::addr_t m_optional_class_methods_addr;
            lldb::addr_t m_instance_properties_addr;
            uint32_t m_size;
            bool m_is_swift;
            bool m_class_only;
            bool m_uses_witness_table;
            bool m_is_error_type;
            
        protected:
            // these come from ProtocolDescriptorFlags in MetadataValues.h
            static const uint32_t SpecialProtocolMask = 0x7F000000U;
            static const uint32_t SpecialProtocolShift = 24;
            
        public:
            Descriptor (SwiftLanguageRuntime& runtime)
            : MetadataUtils (runtime, LLDB_INVALID_ADDRESS)
            { }
            
            Descriptor (SwiftLanguageRuntime& runtime,
                        lldb::addr_t base_descriptor_address);
            
            bool
            IsValid ()
            {
                return m_mangled_name.empty() == false;
            }
            
            lldb::addr_t
            GetISAPlaceholder ()
            {
                return m_isa_placeholder;
            }
            
            std::string
            GetMangledName ()
            {
                return m_mangled_name;
            }
            
            size_t
            GetNumParents ()
            {
                return m_parents.size();
            }
            
            Descriptor
            GetParentAtIndex (size_t idx)
            {
                if (idx >= m_parents.size())
                    return Descriptor(m_runtime);
                return m_parents[idx];
            }
            
            // TODO: do we need accessors for all the ObjC compatibility tables?
            
            bool
            IsSwift ()
            {
                return m_is_swift;
            }
            
            bool
            IsClassOnly ()
            {
                return m_class_only;
            }
            
            bool
            UsesWitnessTable ()
            {
                return m_uses_witness_table;
            }
            
            bool
            IsErrorType ()
            {
                return m_is_error_type;
            }
        };
        
        bool
        IsClassOnly ()
        {
            return m_class_only;
        }
        
        // from the ABI docs: For the "any" types ``protocol<>`` or ``protocol<class>``, this is zero
        bool
        IsAny ()
        {
            return (GetNumProtocols() > 0);
        }
        
        bool
        IsErrorType ();
        
        uint32_t
        GetNumWitnessTables ()
        {
            return m_num_witness_tables;
        }
        
        size_t
        GetNumProtocols ()
        {
            return m_num_protocols;
        }
        
        bool
        IsStaticallyDetermined () override
        {
            return false;
        }
        
        Descriptor
        GetProtocolAtIndex (size_t i);
        
    protected:
        ProtocolMetadata (SwiftLanguageRuntime& runtime,
                          lldb::addr_t base_addr);
        
    private:
        uint32_t m_num_witness_tables;
        bool m_class_only;
        size_t m_num_protocols;
        std::vector <Descriptor> m_protocols;

        DISALLOW_COPY_AND_ASSIGN(ProtocolMetadata);
    };
    
    class MetatypeMetadata : public Metadata
    {
    friend class SwiftLanguageRuntime;
    public:
        static bool classof(const Metadata *S)
        {
            return S->GetKind() == Kind::Metatype;
        }
        
        MetadataSP
        GetInstanceMetadata ();
        
    protected:
        
        MetatypeMetadata (SwiftLanguageRuntime& runtime,
                          lldb::addr_t base_addr);
        
    private:
        MetadataSP m_instance_metadata_sp;
        
        DISALLOW_COPY_AND_ASSIGN(MetatypeMetadata);
    };
    
    class ObjCWrapperMetadata : public NominalTypeMetadata
    {
        friend class SwiftLanguageRuntime;
    public:
        static bool classof(const Metadata *S)
        {
            return S->GetKind() == Kind::ObjCWrapper;
        }
        
        // this won't really be a mangled name
        std::string
        GetMangledName () override;
        
        GenericParameterVector*
        GetGenericParameterVector () override { return nullptr; }
        
        virtual ObjCLanguageRuntime::ClassDescriptorSP
        GetObjectiveCClassDescriptor () 
        {
            return m_objc_class_sp;
        }
        
        bool
        IsStaticallyDetermined () override
        {
            return false;
        }
        
    protected:
        ObjCWrapperMetadata (SwiftLanguageRuntime& runtime,
                             lldb::addr_t base_addr);
        
    private:
        ObjCLanguageRuntime::ClassDescriptorSP m_objc_class_sp;
        
        DISALLOW_COPY_AND_ASSIGN(ObjCWrapperMetadata);
    };
    
    class ClassMetadata : public FieldContainerTypeMetadata
    {
    friend class SwiftLanguageRuntime;
    public:
        static bool classof(const Metadata *S)
        {
            return S->GetKind() == Kind::Class;
        }
        
        std::string
        GetMangledName () override
        {
            return m_mangled_name;
        }
        
        size_t
        GetNumFields () override
        {
            return m_num_fields;
        }
        
        bool
        IsSwiftClass ()
        {
            return ((m_rodata_ptr & 1) == 1);
        }
        
        ObjCLanguageRuntime::ClassDescriptorSP
        GetObjectiveCClassDescriptor ()
        {
            return m_objc_class_desc_sp;
        }
        
        MetadataSP
        GetSuperclassMetadata ()
        {
            return m_superclass_metadata_sp;
        }
        
        size_t
        GetInstanceSize ()
        {
            return m_instance_size;
        }
        
        size_t
        GetInstanceAlignment ()
        {
            return m_instance_align_mask;
        }
        
        Field
        GetFieldAtIndex (size_t) override;
        
        GenericParameterVector*
        GetGenericParameterVector () override
        {
            return m_gpv_ap.get();
        }
        
        bool
        IsObjectiveC ()
        {
            return m_is_objc;
        }
        
        bool
        IsStaticallyDetermined () override
        {
            return false;
        }
        
    protected:
        ClassMetadata (SwiftLanguageRuntime& runtime,
                       lldb::addr_t base_addr);
        
    private:
        bool m_is_objc;
        std::string m_mangled_name;
        lldb::addr_t m_destructor_ptr;
        ObjCLanguageRuntime::ClassDescriptorSP m_objc_class_desc_sp;
        MetadataSP m_superclass_metadata_sp;
        lldb::addr_t m_reserved1; // reserved for ObjC
        lldb::addr_t m_reserved2; // reserved for ObjC
        lldb::addr_t m_rodata_ptr; // we don't read the rodata, but can use the low bit as a tag
        uint32_t m_class_flags;
        uint32_t m_instance_addr_point;
        uint32_t m_instance_size;
        uint16_t m_instance_align_mask;
        uint16_t m_reserved3; // reserved for the runtime
        uint32_t m_class_obj_size;
        size_t m_num_fields;
        std::vector<Field> m_fields;
        std::unique_ptr<GenericParameterVector> m_gpv_ap;

        DISALLOW_COPY_AND_ASSIGN(ClassMetadata);
    };

    class SwiftExceptionPrecondition : public Breakpoint::BreakpointPrecondition
    {
    public:
        SwiftExceptionPrecondition();

        virtual ~SwiftExceptionPrecondition() {}

        bool EvaluatePrecondition(StoppointCallbackContext &context) override;
        void GetDescription(Stream &stream, lldb::DescriptionLevel level) override;
        Error ConfigurePrecondition(Args &args) override;

    protected:
        void AddTypeName(const char *type_name);
        void AddEnumSpec(const char *enum_name, const char *element_name);

    private:
        std::unordered_set<std::string> m_type_names;
        std::unordered_map<std::string, std::vector<std::string> > m_enum_spec;
    };

    virtual
    ~SwiftLanguageRuntime ();
    
    virtual lldb::LanguageType
    GetLanguageType () const override
    {
        return lldb::eLanguageTypeSwift;
    }
    
    void
    ModulesDidLoad (const ModuleList &module_list) override;

    virtual bool
    GetObjectDescription (Stream &str, ValueObject &object) override;
    
    virtual bool
    GetObjectDescription (Stream &str, Value &value, ExecutionContextScope *exe_scope) override;
    
    static bool
    IsSwiftMangledName(const char *name);
    
    struct SwiftErrorDescriptor
    {
    public:
        struct SwiftBridgeableNativeError
        {
        public:
            lldb::addr_t metadata_location;
            lldb::addr_t metadata_ptr_value;
        };

        struct SwiftPureNativeError
        {
        public:
            lldb::addr_t metadata_location;
            lldb::addr_t witness_table_location;
            lldb::addr_t payload_ptr;
        };

        struct SwiftNSError
        {
        public:
            lldb::addr_t instance_ptr_value;
        };
        
        enum class Kind
        {
            eSwiftBridgeableNative,
            eSwiftPureNative,
            eBridged,
            eNotAnError
        };
        
        Kind m_kind;
        SwiftBridgeableNativeError m_bridgeable_native;
        SwiftPureNativeError m_pure_native;
        SwiftNSError m_bridged;
        
        operator bool ()
        {
            return m_kind != Kind::eNotAnError;
        }
        
        SwiftErrorDescriptor();
        
        SwiftErrorDescriptor(const SwiftErrorDescriptor &rhs) = default;
    };
    
    // provide a quick and yet somewhat reasonable guess as to whether
    // this ValueObject represents something that validly conforms
    // to the magic ErrorType protocol
    virtual bool
    IsValidErrorValue (ValueObject& in_value,
                       SwiftErrorDescriptor *out_error_descriptor = nullptr);
    
    virtual lldb::BreakpointResolverSP
    CreateExceptionResolver (Breakpoint *bkpt, bool catch_bp, bool throw_bp) override;

    SwiftExceptionPrecondition *
    GetExceptionPrecondition();

    static lldb::ValueObjectSP
    CalculateErrorValueFromFirstArgument(lldb::StackFrameSP frame_sp, ConstString name);

    lldb::ValueObjectSP
    CalculateErrorValueObjectAtAddress (lldb::addr_t addr, ConstString name, bool persistent);
    
    lldb::addr_t
    GetErrorReturnLocationForFrame(lldb::StackFrameSP frame_sp);
    
    static void
    RegisterGlobalError(Target &target, ConstString name, lldb::addr_t addr);

    // If you are at the initial instruction of the frame passed in, then this will examine the call
    // arguments, and if any of them is a function pointer, this will push the address of the function
    // into addresses.  If debug_only is true, then it will only push function pointers that are in user
    // code.
    
    void
    FindFunctionPointersInCall (StackFrame &frame,
                                std::vector<Address> &addresses,
                                bool debug_only = true,
                                bool resolve_thunks = true);
    
    virtual lldb::ThreadPlanSP
    GetStepThroughTrampolinePlan (Thread &thread, bool stop_others);
    
    bool
    IsSymbolARuntimeThunk (const Symbol &symbol) override;

    // in some cases, compilers will output different names for one same type. when tht happens, it might be impossible
    // to construct SBType objects for a valid type, because the name that is available is not the same as the name that
    // can be used as a search key in FindTypes(). the equivalents map here is meant to return possible alternative names
    // for a type through which a search can be conducted. Currently, this is only enabled for C++ but can be extended
    // to ObjC or other languages if necessary
    static uint32_t
    FindEquivalentNames(ConstString type_name, std::vector<ConstString>& equivalents);
  
    // this call should return true if it could set the name and/or the type
    virtual bool
    GetDynamicTypeAndAddress (ValueObject &in_value,
                              lldb::DynamicValueType use_dynamic,
                              TypeAndOrName &class_type_or_name,
                              Address &address,
                              Value::ValueType &value_type) override;
    
    virtual TypeAndOrName
    FixUpDynamicType(const TypeAndOrName& type_and_or_name,
                     ValueObject& static_value) override;
    
    bool
    IsRuntimeSupportValue (ValueObject& valobj) override;

    virtual CompilerType
    DoArchetypeBindingForType (StackFrame& stack_frame,
                               CompilerType base_type,
                               SwiftASTContext *ast_context = nullptr);

    virtual CompilerType
    GetConcreteType (ExecutionContextScope *exe_scope,
                     ConstString abstract_type_name) override;

    virtual bool
    CouldHaveDynamicValue (ValueObject &in_value) override;
    
    // TODO: one should have a better entry point for metadata retrieval than an address
    virtual MetadataSP
    GetMetadataForLocation (lldb::addr_t addr);
    
    virtual MetadataSP
    GetMetadataForType (CompilerType type);
    
    virtual CompilerType
    GetTypeForMetadata (MetadataSP metadata_sp,
                        SwiftASTContext * swift_ast_ctx,
                        Error& error);
    
    virtual GenericPatternSP
    GetGenericPatternForType (CompilerType type);

    virtual NominalTypeDescriptorSP
    GetNominalTypeDescriptorForType (CompilerType type);
    
    virtual NominalTypeDescriptorSP
    GetNominalTypeDescriptorForLocation (lldb::addr_t addr);

    void
    AddToLibraryNegativeCache (const char *library_name);
    
    bool
    IsInLibraryNegativeCache (const char *library_name);
    
    // Swift uses a few known-unused bits in ObjC pointers
    // to record useful-for-bridging information
    // This API's task is to return such pointer+info aggregates
    // back to a pure pointer
    lldb::addr_t
    MaskMaybeBridgedPointer (lldb::addr_t,
                             lldb::addr_t * = nullptr);

    ConstString
    GetErrorBackstopName ();

    ConstString
    GetStandardLibraryName();
    
    virtual bool
    GetReferenceCounts (ValueObject& valobj, size_t &strong, size_t &weak);


protected:
    //------------------------------------------------------------------
    // Classes that inherit from SwiftLanguageRuntime can see and modify these
    //------------------------------------------------------------------
    SwiftLanguageRuntime(Process *process);
    
    Value::ValueType
    GetValueType (Value::ValueType static_value_type,
                  const CompilerType& static_type,
                  const CompilerType& dynamic_type);
    
    virtual bool
    GetDynamicTypeAndAddress_Class (ValueObject &in_value,
                                    lldb::DynamicValueType use_dynamic,
                                    TypeAndOrName &class_type_or_name,
                                    Address &address);

    virtual bool
    GetDynamicTypeAndAddress_Protocol (ValueObject &in_value,
                                       lldb::DynamicValueType use_dynamic,
                                       TypeAndOrName &class_type_or_name,
                                       Address &address);

    virtual bool
    GetDynamicTypeAndAddress_ErrorType (ValueObject &in_value,
                                        lldb::DynamicValueType use_dynamic,
                                        TypeAndOrName &class_type_or_name,
                                        Address &address);
    
    virtual bool
    GetDynamicTypeAndAddress_Archetype (ValueObject &in_value,
                                        lldb::DynamicValueType use_dynamic,
                                        TypeAndOrName &class_type_or_name,
                                        Address &address);

    virtual bool
    GetDynamicTypeAndAddress_Tuple (ValueObject &in_value,
                                    lldb::DynamicValueType use_dynamic,
                                    TypeAndOrName &class_type_or_name,
                                    Address &address);
    
    virtual bool
    GetDynamicTypeAndAddress_Struct (ValueObject &in_value,
                                     lldb::DynamicValueType use_dynamic,
                                     TypeAndOrName &class_type_or_name,
                                     Address &address);
    
    virtual bool
    GetDynamicTypeAndAddress_Enum (ValueObject &in_value,
                                   lldb::DynamicValueType use_dynamic,
                                   TypeAndOrName &class_type_or_name,
                                   Address &address);

    virtual bool
    GetDynamicTypeAndAddress_IndirectEnumCase (ValueObject &in_value,
                                               lldb::DynamicValueType use_dynamic,
                                               TypeAndOrName &class_type_or_name,
                                               Address &address);
    
    virtual bool
    GetDynamicTypeAndAddress_Metadata (ValueObject &in_value,
                                       MetadataSP metadata_sp,
                                       lldb::DynamicValueType use_dynamic,
                                       TypeAndOrName &class_type_or_name,
                                       Address &address);
    
    virtual MetadataSP
    GetMetadataForTypeNameAndFrame (const char* type_name,
                                    StackFrame* frame);

    bool
    GetTargetOfPartialApply (CompileUnit &cu, ConstString &apply_name, SymbolContext &sc);

    AppleObjCRuntimeV2*
    GetObjCRuntime ();
    
    void
    SetupSwiftObject ();
    
    void
    SetupSwiftError ();
    
    bool
    LoadDumpForDebugger (Error& error);
    
    const CompilerType&
    GetClassMetadataType ();
    
    const CompilerType&
    GetNominalTypeDescriptorType ();
    
    const CompilerType&
    GetBoxMetadataType ();
    
    // the Swift runtime assumes that metadata will not go away, so caching
    // by address is a reasonable strategy
    std::map<lldb::addr_t, MetadataSP> m_metadata_cache;
    
    std::map<lldb::addr_t, GenericPatternSP> m_generic_pattern_cache;
    
    std::map<CompilerType, NominalTypeDescriptorSP> m_nominal_descriptor_cache;
    
    CompilerType m_class_metadata_type;
    CompilerType m_nominal_type_descriptor_type;
    CompilerType m_box_metadata_type;
    std::unordered_set<std::string> m_library_negative_cache;  // We have to load swift dependent libraries by hand,
    Mutex                           m_negative_cache_mutex;    // but if they are missing, we shouldn't keep trying.
    
    LazyBool m_loaded_DumpForDebugger;
    
    llvm::Optional<lldb::addr_t> m_SwiftNativeNSErrorISA;

private:
    DISALLOW_COPY_AND_ASSIGN (SwiftLanguageRuntime);
};
    
} // namespace lldb_private

#endif  // liblldb_SwiftLanguageRuntime_h_
