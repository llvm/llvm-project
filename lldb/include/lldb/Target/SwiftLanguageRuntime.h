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
#include <mutex>
#include <tuple>
#include <vector>
// Other libraries and framework includes
// Project includes
#include "lldb/Core/PluginInterface.h"
#include "lldb/lldb-private.h"
#include "Plugins/LanguageRuntime/ObjC/AppleObjCRuntime/AppleObjCRuntimeV2.h"
#include "lldb/Target/LanguageRuntime.h"

#include "llvm/Support/Casting.h"
#include "llvm/ADT/Optional.h"

namespace swift {
    namespace remote {
        class MemoryReader;
        class RemoteAddress;
    }
    namespace remoteAST {
        class RemoteASTContext;
    }
    enum class MetadataKind : uint32_t;
    class TypeBase;
}

namespace lldb_private {
    
class SwiftLanguageRuntime :
    public LanguageRuntime
{
public:
    class MetadataPromise;
    typedef std::shared_ptr<MetadataPromise> MetadataPromiseSP;
    
    class MemberVariableOffsetResolver;
    typedef std::shared_ptr<MemberVariableOffsetResolver> MemberVariableOffsetResolverSP;
    
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
    
    class MetadataPromise
    {
        friend class SwiftLanguageRuntime;
        
        MetadataPromise (swift::ASTContext *,
                         SwiftLanguageRuntime *,
                         lldb::addr_t);
        
        swift::ASTContext *m_swift_ast;
        std::unique_ptr<swift::remoteAST::RemoteASTContext> m_remote_ast;
        SwiftLanguageRuntime *m_swift_runtime;
        lldb::addr_t m_metadata_location;
        llvm::Optional<swift::MetadataKind> m_metadata_kind;
        llvm::Optional<CompilerType> m_compiler_type;
        
    public:
        CompilerType
        FulfillTypePromise (Error *error = nullptr);
        
        llvm::Optional<swift::MetadataKind>
        FulfillKindPromise (Error *error = nullptr);
        
        bool
        IsStaticallyDetermined ();
    };
    
    class MemberVariableOffsetResolver
    {
        friend class SwiftLanguageRuntime;
        
        MemberVariableOffsetResolver(swift::ASTContext *,
                                     SwiftLanguageRuntime *,
                                     swift::TypeBase *);
        
        swift::ASTContext *m_swift_ast;
        std::unique_ptr<swift::remoteAST::RemoteASTContext> m_remote_ast;
        SwiftLanguageRuntime *m_swift_runtime;
        swift::TypeBase *m_swift_type;
        std::unordered_map<const char*, uint64_t> m_offsets;
        
    public:
        llvm::Optional<uint64_t>
        ResolveOffset (ValueObject *valobj,
                       ConstString ivar_name,
                       Error* = nullptr);
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
    
    virtual MetadataPromiseSP
    GetMetadataPromise (lldb::addr_t addr,
                        SwiftASTContext* swift_ast_ctx = nullptr);

    virtual MemberVariableOffsetResolverSP
    GetMemberVariableOffsetResolver (CompilerType compiler_type);
    
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
    
    // Swift uses a few known-unused bits in weak,unowned,unmanaged references
    // to record useful runtime information
    // This API's task is to strip those bits if necessary and return
    // a pure pointer (or a tagged pointer)
    lldb::addr_t
    MaybeMaskNonTrivialReferencePointer (lldb::addr_t);

    ConstString
    GetErrorBackstopName ();

    ConstString
    GetStandardLibraryName();
    
    ConstString
    GetStandardLibraryBaseName();
    
    virtual bool
    GetReferenceCounts (ValueObject& valobj, size_t &strong, size_t &weak);

    lldb::SyntheticChildrenSP
    GetBridgedSyntheticChildProvider (ValueObject& valobj);
    
protected:
    //------------------------------------------------------------------
    // Classes that inherit from SwiftLanguageRuntime can see and modify these
    //------------------------------------------------------------------
    SwiftLanguageRuntime(Process *process);
    
    Value::ValueType
    GetValueType (Value::ValueType static_value_type,
                  const CompilerType& static_type,
                  const CompilerType& dynamic_type,
                  bool is_indirect_enum_case);
    
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
    GetDynamicTypeAndAddress_Promise (ValueObject &in_value,
                                      MetadataPromiseSP promise_sp,
                                      lldb::DynamicValueType use_dynamic,
                                      TypeAndOrName &class_type_or_name,
                                      Address &address);
    
    virtual MetadataPromiseSP
    GetPromiseForTypeNameAndFrame (const char* type_name,
                                   StackFrame* frame);
    
    bool
    GetTargetOfPartialApply (CompileUnit &cu, ConstString &apply_name, SymbolContext &sc);

    AppleObjCRuntimeV2*
    GetObjCRuntime ();
    
    void
    SetupSwiftError ();
    
    const CompilerType&
    GetBoxMetadataType ();
    
    std::shared_ptr<swift::remote::MemoryReader>
    GetMemoryReader ();
    
    SwiftASTContext*
    GetScratchSwiftASTContext ();
    
    std::unordered_set<std::string> m_library_negative_cache;  // We have to load swift dependent libraries by hand,
    std::mutex                      m_negative_cache_mutex;    // but if they are missing, we shouldn't keep trying.
    
    llvm::Optional<lldb::addr_t> m_SwiftNativeNSErrorISA;

    std::shared_ptr<swift::remote::MemoryReader> m_memory_reader_sp;
    
    template <typename Key1, typename Key2, typename Value1>
    struct KeyHasher
    {
        using KeyType = std::tuple<Key1,Key2>;
        using HasherType = KeyHasher<Key1,Key2,Value1>;
        using MapType = std::unordered_map<KeyType, Value1, HasherType>;

        size_t
        operator()(const std::tuple<Key1,Key2> &key) const
        {
            // fairly trivial hash combiner function
            auto hash1 = std::hash<Key1>()(std::get<0>(key));
            auto hash2 = std::hash<Key2>()(std::get<1>(key));
            return hash2 + 0x9e3779b9 + (hash1<<6) + (hash1>>2);
        }
    };

    typename KeyHasher<swift::ASTContext*, lldb::addr_t, MetadataPromiseSP>::MapType m_promises_map;
    typename KeyHasher<swift::ASTContext*, swift::TypeBase*, MemberVariableOffsetResolverSP>::MapType m_resolvers_map;

    std::unordered_map<const char*, lldb::SyntheticChildrenSP> m_bridged_synthetics_map;
    
    CompilerType m_box_metadata_type;
    
private:
    DISALLOW_COPY_AND_ASSIGN (SwiftLanguageRuntime);
};
    
} // namespace lldb_private

#endif  // liblldb_SwiftLanguageRuntime_h_
