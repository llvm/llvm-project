////////////////////////////////////////////////////////////////////////////////
// Note: This file is a work in progress. Please do not apply non-trivial
// updates unless you have talked to Sean Hunt <rideau3@gmail.com> prior.
// Merely adding a new attribute is a trivial update.
////////////////////////////////////////////////////////////////////////////////

// An attribute's subject is whatever it appertains to. In this file, it is
// more accurately a list of things that an attribute can appertain to. All
// Decls and Stmts are possibly AttrSubjects (even though the syntax may not
// allow attributes on a given Decl or Stmt).
class AttrSubject;

include "clang/Basic/DeclNodes.td"
include "clang/Basic/StmtNodes.td"

// A subset-subject is an AttrSubject constrained to operate only on some subset
// of that subject.
//
// The description is used in output messages to specify what the subject
// represents. FIXME: Deal with translation issues.
//
// The code fragment is a boolean expression that will confirm that the subject
// meets the requirements; the subject will have the name S, and will have the
// type specified by the base. It should be a simple boolean expression.
class SubsetSubject<AttrSubject base, string description, code check>
    : AttrSubject {
  AttrSubject Base = base;
  string Description = description;
  code CheckCode = check;
}

// This is the type of a variable which C++0x defines [[aligned()]] as being
// a possible subject.
def NormalVar : SubsetSubject<Var, "non-register, non-parameter variable",
                              [{S->getStorageClass() != VarDecl::Register &&
                                S->getKind() != Decl::ImplicitParam &&
                                S->getKind() != Decl::ParmVar &&
                                S->getKind() != Decl::NonTypeTemplateParm}]>;
def CXXVirtualMethod : SubsetSubject<CXXRecord, "virtual member function",
                                     [{S->isVirtual()}]>;
def NonBitField : SubsetSubject<Field, "non-bit field",
                                [{!S->isBitField()}]>;

// A single argument to an attribute
class Argument<string name> {
  string Name = name;
}

class BoolArgument<string name> : Argument<name>;
class IdentifierArgument<string name> : Argument<name>;
class IntArgument<string name> : Argument<name>;
class StringArgument<string name> : Argument<name>;
class ExprArgument<string name> : Argument<name>;
class FunctionArgument<string name> : Argument<name>;
class TypeArgument<string name> : Argument<name>;
class UnsignedArgument<string name> : Argument<name>;
class SourceLocArgument<string name> : Argument<name>;
class VariadicUnsignedArgument<string name> : Argument<name>;
class VariadicExprArgument<string name> : Argument<name>;

// A version of the form major.minor[.subminor].
class VersionArgument<string name> : Argument<name>;

// This one's a doozy, so it gets its own special type
// It can be an unsigned integer, or a type. Either can
// be dependent.
class AlignedArgument<string name> : Argument<name>;

// An integer argument with a default value
class DefaultIntArgument<string name, int default> : IntArgument<name> {
  int Default = default;
}

// This argument is more complex, it includes the enumerator type name,
// a list of strings to accept, and a list of enumerators to map them to.
class EnumArgument<string name, string type, list<string> values,
                         list<string> enums> : Argument<name> {
  string Type = type;
  list<string> Values = values;
  list<string> Enums = enums;
}

// This handles one spelling of an attribute.
class Spelling<string name, string variety> {
  string Name = name;
  string Variety = variety;
}

class GNU<string name> : Spelling<name, "GNU">;
class Declspec<string name> : Spelling<name, "Declspec">;
class CXX11<string namespace, string name> : Spelling<name, "CXX11"> {
  string Namespace = namespace;
}

class Attr {
  // The various ways in which an attribute can be spelled in source
  list<Spelling> Spellings;
  // The things to which an attribute can appertain
  list<AttrSubject> Subjects;
  // The arguments allowed on an attribute
  list<Argument> Args = [];
  // Set to true for attributes with arguments which require delayed parsing. 
  bit LateParsed = 0;  
  // Set to false to prevent an attribute from being propagated from a template
  // to the instantiation.
  bit Clone = 1;
  // Set to true for attributes which must be instantiated within templates
  bit TemplateDependent = 0;
  // Set to true for attributes that have a corresponding AST node.
  bit ASTNode = 1;
  // Set to true for attributes which have handler in Sema.
  bit SemaHandler = 1;
  // Set to true for attributes that are completely ignored.
  bit Ignored = 0;
  // Set to true if each of the spellings is a distinct attribute.
  bit DistinctSpellings = 0;
  // Any additional text that should be included verbatim in the class.  
  code AdditionalMembers = [{}];
}

/// An inheritable attribute is inherited by later redeclarations.
class InheritableAttr : Attr;

/// An inheritable parameter attribute is inherited by later
/// redeclarations, even when it's written on a parameter.
class InheritableParamAttr : InheritableAttr;

//
// Attributes begin here
//

def AddressSpace : Attr {
  let Spellings = [GNU<"address_space">];
  let Args = [IntArgument<"AddressSpace">];
  let ASTNode = 0;  
}

def Alias : InheritableAttr {
  let Spellings = [GNU<"alias">];
  let Args = [StringArgument<"Aliasee">];
}

def Aligned : InheritableAttr {
  let Spellings = [GNU<"aligned">, GNU<"align">];
  let Subjects = [NonBitField, NormalVar, Tag];
  let Args = [AlignedArgument<"Alignment">, BoolArgument<"IsMSDeclSpec">];
}

def AlignMac68k : InheritableAttr {
  let Spellings = [];
  let SemaHandler = 0;
}

def AllocSize : Attr {
  let Spellings = [GNU<"alloc_size">];
  let Args = [VariadicUnsignedArgument<"Args">];
}

def AlwaysInline : InheritableAttr {
  let Spellings = [GNU<"always_inline">];
}

def TLSModel : InheritableAttr {
  let Spellings = [GNU<"tls_model">];
  let Subjects = [Var];
  let Args = [StringArgument<"Model">];
}

def AnalyzerNoReturn : InheritableAttr {
  let Spellings = [GNU<"analyzer_noreturn">];
}

def Annotate : InheritableParamAttr {
  let Spellings = [GNU<"annotate">];
  let Args = [StringArgument<"Annotation">];
}

def AsmLabel : InheritableAttr {
  let Spellings = [];
  let Args = [StringArgument<"Label">];
  let SemaHandler = 0;
}

def Availability : InheritableAttr {
  let Spellings = [GNU<"availability">];
  let Args = [IdentifierArgument<"platform">, VersionArgument<"introduced">,
              VersionArgument<"deprecated">, VersionArgument<"obsoleted">,
              BoolArgument<"unavailable">, StringArgument<"message">];
  let AdditionalMembers =
[{static llvm::StringRef getPrettyPlatformName(llvm::StringRef Platform) {
    return llvm::StringSwitch<llvm::StringRef>(Platform)
             .Case("ios", "iOS")
             .Case("macosx", "OS X")
             .Default(llvm::StringRef());
} }];
}

def Blocks : InheritableAttr {
  let Spellings = [GNU<"blocks">];
  let Args = [EnumArgument<"Type", "BlockType", ["byref"], ["ByRef"]>];
}

def Bounded : Attr {
  let Spellings = [GNU<"bounded">];
  let ASTNode = 0;
  let SemaHandler = 0;
  let Ignored = 1;
}

def CarriesDependency : InheritableParamAttr {
  let Spellings = [GNU<"carries_dependency">, CXX11<"","carries_dependency">,
                   CXX11<"std","carries_dependency">];
  let Subjects = [ParmVar, Function];
}

def CDecl : InheritableAttr {
  let Spellings = [GNU<"cdecl">, GNU<"__cdecl">];
}

// cf_audited_transfer indicates that the given function has been
// audited and has been marked with the appropriate cf_consumed and
// cf_returns_retained attributes.  It is generally applied by
// '#pragma clang arc_cf_code_audited' rather than explicitly.
def CFAuditedTransfer : InheritableAttr {
  let Spellings = [GNU<"cf_audited_transfer">];
  let Subjects = [Function];
}

// cf_unknown_transfer is an explicit opt-out of cf_audited_transfer.
// It indicates that the function has unknown or unautomatable
// transfer semantics.
def CFUnknownTransfer : InheritableAttr {
  let Spellings = [GNU<"cf_unknown_transfer">];
  let Subjects = [Function];
}

def CFReturnsAutoreleased : Attr {
  let Spellings = [GNU<"cf_returns_autoreleased">];
  let ASTNode = 0;
}

def CFReturnsRetained : InheritableAttr {
  let Spellings = [GNU<"cf_returns_retained">];
  let Subjects = [ObjCMethod, Function];
}

def CFReturnsNotRetained : InheritableAttr {
  let Spellings = [GNU<"cf_returns_not_retained">];
  let Subjects = [ObjCMethod, Function];
}

def CFConsumed : InheritableParamAttr {
  let Spellings = [GNU<"cf_consumed">];
  let Subjects = [ParmVar];
}

def Cleanup : InheritableAttr {
  let Spellings = [GNU<"cleanup">];
  let Args = [FunctionArgument<"FunctionDecl">];
}

def Cold : InheritableAttr {
  let Spellings = [GNU<"cold">];
}

def Common : InheritableAttr {
  let Spellings = [GNU<"common">];
}

def Const : InheritableAttr {
  let Spellings = [GNU<"const">, GNU<"__const">];
}

def Constructor : InheritableAttr {
  let Spellings = [GNU<"constructor">];
  let Args = [IntArgument<"Priority">];
}

def CUDAConstant : InheritableAttr {
  let Spellings = [GNU<"constant">];
}

def CUDADevice : InheritableAttr {
  let Spellings = [GNU<"device">];
}

def CUDAGlobal : InheritableAttr {
  let Spellings = [GNU<"global">];
}

def CUDAHost : InheritableAttr {
  let Spellings = [GNU<"host">];
}

def CUDALaunchBounds : InheritableAttr {
  let Spellings = [GNU<"launch_bounds">];
  let Args = [IntArgument<"MaxThreads">, DefaultIntArgument<"MinBlocks", 0>];
}

def CUDAShared : InheritableAttr {
  let Spellings = [GNU<"shared">];
}

def OpenCLKernel : Attr {
  let Spellings = [GNU<"opencl_kernel_function">];
}

def OpenCLImageAccess : Attr {
  let Spellings = [GNU<"opencl_image_access">];
  let Args = [IntArgument<"Access">];
  let ASTNode = 0;
}

def Deprecated : InheritableAttr {
  let Spellings = [GNU<"deprecated">];
  let Args = [StringArgument<"Message">];
}

def Destructor : InheritableAttr {
  let Spellings = [GNU<"destructor">];
  let Args = [IntArgument<"Priority">];
}

def ExtVectorType : Attr {
  let Spellings = [GNU<"ext_vector_type">];
  let Args = [ExprArgument<"NumElements">];
  let ASTNode = 0;
}

def FallThrough : Attr {
  let Spellings = [CXX11<"clang","fallthrough">];
  let Subjects = [NullStmt];
}

def FastCall : InheritableAttr {
  let Spellings = [GNU<"fastcall">, GNU<"__fastcall">];
}

def Final : InheritableAttr {
  let Spellings = [];
  let SemaHandler = 0;
}

def MinSize : InheritableAttr {
  let Spellings = [GNU<"minsize">];
  let Subjects = [Function];
}

def Format : InheritableAttr {
  let Spellings = [GNU<"format">];
  let Args = [StringArgument<"Type">, IntArgument<"FormatIdx">,
              IntArgument<"FirstArg">];
}

def FormatArg : InheritableAttr {
  let Spellings = [GNU<"format_arg">];
  let Args = [IntArgument<"FormatIdx">];
}

def GNUInline : InheritableAttr {
  let Spellings = [GNU<"gnu_inline">];
}

def Hot : InheritableAttr {
  let Spellings = [GNU<"hot">];
}

def IBAction : InheritableAttr {
  let Spellings = [GNU<"ibaction">];
}

def IBOutlet : InheritableAttr {
  let Spellings = [GNU<"iboutlet">];
}

def IBOutletCollection : InheritableAttr {
  let Spellings = [GNU<"iboutletcollection">];
  let Args = [TypeArgument<"Interface">, SourceLocArgument<"InterfaceLoc">];
}

def Malloc : InheritableAttr {
  let Spellings = [GNU<"malloc">];
}

def MaxFieldAlignment : InheritableAttr {
  let Spellings = [];
  let Args = [UnsignedArgument<"Alignment">];
  let SemaHandler = 0;
}

def MayAlias : InheritableAttr {
  let Spellings = [GNU<"may_alias">];
}

def MSP430Interrupt : InheritableAttr {
  let Spellings = [];
  let Args = [UnsignedArgument<"Number">];
  let SemaHandler = 0;
}

def MBlazeInterruptHandler : InheritableAttr {
  let Spellings = [];
  let SemaHandler = 0;
}

def MBlazeSaveVolatiles : InheritableAttr {
  let Spellings = [];
  let SemaHandler = 0;
}

def Mode : Attr {
  let Spellings = [GNU<"mode">];
  let Args = [IdentifierArgument<"Mode">];
  let ASTNode = 0;
}

def Naked : InheritableAttr {
  let Spellings = [GNU<"naked">];
}

def NeonPolyVectorType : Attr {
  let Spellings = [GNU<"neon_polyvector_type">];
  let Args = [IntArgument<"NumElements">];
  let ASTNode = 0;
}

def NeonVectorType : Attr {
  let Spellings = [GNU<"neon_vector_type">];
  let Args = [IntArgument<"NumElements">];
  let ASTNode = 0;
}

def ReturnsTwice : InheritableAttr {
  let Spellings = [GNU<"returns_twice">];
}

def NoCommon : InheritableAttr {
  let Spellings = [GNU<"nocommon">];
}

def NoDebug : InheritableAttr {
  let Spellings = [GNU<"nodebug">];
}

def NoInline : InheritableAttr {
  let Spellings = [GNU<"noinline">];
}

def NonNull : InheritableAttr {
  let Spellings = [GNU<"nonnull">];
  let Args = [VariadicUnsignedArgument<"Args">];
  let AdditionalMembers =
[{bool isNonNull(unsigned idx) const {
    for (args_iterator i = args_begin(), e = args_end();
         i != e; ++i)
      if (*i == idx)
        return true;
    return false;
  } }];
}

def NoReturn : InheritableAttr {
  let Spellings = [GNU<"noreturn">, CXX11<"","noreturn">,
                   CXX11<"std","noreturn">];
  // FIXME: Does GCC allow this on the function instead?
  let Subjects = [Function];
}

def NoInstrumentFunction : InheritableAttr {
  let Spellings = [GNU<"no_instrument_function">];
  let Subjects = [Function];
}

def NoThrow : InheritableAttr {
  let Spellings = [GNU<"nothrow">];
}

def NSBridged : InheritableAttr {
  let Spellings = [GNU<"ns_bridged">];
  let Subjects = [Record];
  let Args = [IdentifierArgument<"BridgedType">];
}

def NSReturnsRetained : InheritableAttr {
  let Spellings = [GNU<"ns_returns_retained">];
  let Subjects = [ObjCMethod, Function];
}

def NSReturnsNotRetained : InheritableAttr {
  let Spellings = [GNU<"ns_returns_not_retained">];
  let Subjects = [ObjCMethod, Function];
}

def NSReturnsAutoreleased : InheritableAttr {
  let Spellings = [GNU<"ns_returns_autoreleased">];
  let Subjects = [ObjCMethod, Function];
}

def NSConsumesSelf : InheritableAttr {
  let Spellings = [GNU<"ns_consumes_self">];
  let Subjects = [ObjCMethod];
}

def NSConsumed : InheritableParamAttr {
  let Spellings = [GNU<"ns_consumed">];
  let Subjects = [ParmVar];
}

def ObjCException : InheritableAttr {
  let Spellings = [GNU<"objc_exception">];
}

def ObjCMethodFamily : InheritableAttr {
  let Spellings = [GNU<"objc_method_family">];
  let Subjects = [ObjCMethod];
  let Args = [EnumArgument<"Family", "FamilyKind",
               ["none", "alloc", "copy", "init", "mutableCopy", "new"],
               ["OMF_None", "OMF_alloc", "OMF_copy", "OMF_init",
                "OMF_mutableCopy", "OMF_new"]>];
}

def ObjCNSObject : InheritableAttr {
  let Spellings = [GNU<"NSObject">];
}

def ObjCPreciseLifetime : Attr {
  let Spellings = [GNU<"objc_precise_lifetime">];
  let Subjects = [Var];
}

def ObjCReturnsInnerPointer : Attr {
  let Spellings = [GNU<"objc_returns_inner_pointer">];
  let Subjects = [ObjCMethod];
}

def ObjCRequiresSuper : InheritableAttr {
  let Spellings = [GNU<"objc_requires_super">];
  let Subjects = [ObjCMethod];
}

def ObjCRootClass : Attr {
  let Spellings = [GNU<"objc_root_class">];
  let Subjects = [ObjCInterface];
}

def Overloadable : Attr {
  let Spellings = [GNU<"overloadable">];
}

def Override : InheritableAttr { 
  let Spellings = [];
  let SemaHandler = 0;
}

def Ownership : InheritableAttr {
  let Spellings = [GNU<"ownership_holds">, GNU<"ownership_returns">,
                   GNU<"ownership_takes">];
  let DistinctSpellings = 1;
  let Args = [EnumArgument<"OwnKind", "OwnershipKind",
                    ["ownership_holds", "ownership_returns", "ownership_takes"],
                    ["Holds", "Returns", "Takes"]>,
              StringArgument<"Module">, VariadicUnsignedArgument<"Args">];
}

def Packed : InheritableAttr {
  let Spellings = [GNU<"packed">];
}

def PnaclCall : InheritableAttr {
  let Spellings = [GNU<"pnaclcall">];
}

def Pcs : InheritableAttr {
  let Spellings = [GNU<"pcs">];
  let Args = [EnumArgument<"PCS", "PCSType",
                           ["aapcs", "aapcs-vfp"],
                           ["AAPCS", "AAPCS_VFP"]>];
}

def Pure : InheritableAttr {
  let Spellings = [GNU<"pure">];
}

def Regparm : InheritableAttr {
  let Spellings = [GNU<"regparm">];
  let Args = [UnsignedArgument<"NumParams">];
}

def ReqdWorkGroupSize : InheritableAttr {
  let Spellings = [GNU<"reqd_work_group_size">];
  let Args = [UnsignedArgument<"XDim">, UnsignedArgument<"YDim">,
              UnsignedArgument<"ZDim">];
}

def WorkGroupSizeHint :  InheritableAttr {
  let Spellings = [GNU<"work_group_size_hint">];
  let Args = [UnsignedArgument<"XDim">, 
              UnsignedArgument<"YDim">,
              UnsignedArgument<"ZDim">];
}

def InitPriority : InheritableAttr {
  let Spellings = [GNU<"init_priority">];
  let Args = [UnsignedArgument<"Priority">];
}

def Section : InheritableAttr {
  let Spellings = [GNU<"section">];
  let Args = [StringArgument<"Name">];
}

def Sentinel : InheritableAttr {
  let Spellings = [GNU<"sentinel">];
  let Args = [DefaultIntArgument<"Sentinel", 0>,
              DefaultIntArgument<"NullPos", 0>];
}

def StdCall : InheritableAttr {
  let Spellings = [GNU<"stdcall">, GNU<"__stdcall">];
}

def ThisCall : InheritableAttr {
  let Spellings = [GNU<"thiscall">, GNU<"__thiscall">];
}

def Pascal : InheritableAttr {
  let Spellings = [GNU<"pascal">];
}

def TransparentUnion : InheritableAttr {
  let Spellings = [GNU<"transparent_union">];
}

def Unavailable : InheritableAttr {
  let Spellings = [GNU<"unavailable">];
  let Args = [StringArgument<"Message">];
}

def ArcWeakrefUnavailable : InheritableAttr {
  let Spellings = [GNU<"objc_arc_weak_reference_unavailable">];
  let Subjects = [ObjCInterface];
}

def ObjCGC : Attr {
  let Spellings = [GNU<"objc_gc">];
  let Args = [IdentifierArgument<"Kind">];
  let ASTNode = 0;
}

def ObjCOwnership : Attr {
  let Spellings = [GNU<"objc_ownership">];
  let Args = [IdentifierArgument<"Kind">];
  let ASTNode = 0;
}

def ObjCRequiresPropertyDefs : InheritableAttr {
  let Spellings = [GNU<"objc_requires_property_definitions">];
  let Subjects = [ObjCInterface];
}

def Unused : InheritableAttr {
  let Spellings = [GNU<"unused">];
}

def Used : InheritableAttr {
  let Spellings = [GNU<"used">];
}

def Uuid : InheritableAttr {
  let Spellings = [GNU<"uuid">];
  let Args = [StringArgument<"Guid">];
  let Subjects = [CXXRecord];
}

def VectorSize : Attr {
  let Spellings = [GNU<"vector_size">];
  let Args = [ExprArgument<"NumBytes">];
  let ASTNode = 0;
}

def VecTypeHint : Attr {
  let Spellings = [GNU<"vec_type_hint">];
  let ASTNode = 0;
  let SemaHandler = 0;
  let Ignored = 1;
}

def Visibility : InheritableAttr {
  let Clone = 0;
  let Spellings = [GNU<"visibility">];
  let Args = [EnumArgument<"Visibility", "VisibilityType",
                           ["default", "hidden", "internal", "protected"],
                           ["Default", "Hidden", "Hidden", "Protected"]>];
}

def VecReturn : InheritableAttr {
  let Spellings = [GNU<"vecreturn">];
  let Subjects = [CXXRecord];
}

def WarnUnusedResult : InheritableAttr {
  let Spellings = [GNU<"warn_unused_result">];
}

def Weak : InheritableAttr {
  let Spellings = [GNU<"weak">];
}

def WeakImport : InheritableAttr {
  let Spellings = [GNU<"weak_import">];
}

def WeakRef : InheritableAttr {
  let Spellings = [GNU<"weakref">];
}

def X86ForceAlignArgPointer : InheritableAttr {
  let Spellings = [];
}

// AddressSafety attribute (e.g. for AddressSanitizer)
def NoAddressSafetyAnalysis : InheritableAttr {
  let Spellings = [GNU<"no_address_safety_analysis">];
}

// C/C++ Thread safety attributes (e.g. for deadlock, data race checking)

def GuardedVar : InheritableAttr {
  let Spellings = [GNU<"guarded_var">];
}

def PtGuardedVar : InheritableAttr {
  let Spellings = [GNU<"pt_guarded_var">];
}

def Lockable : InheritableAttr {
  let Spellings = [GNU<"lockable">];
}

def ScopedLockable : InheritableAttr {
  let Spellings = [GNU<"scoped_lockable">];
}

def NoThreadSafetyAnalysis : InheritableAttr {
  let Spellings = [GNU<"no_thread_safety_analysis">];
}

def GuardedBy : InheritableAttr {
  let Spellings = [GNU<"guarded_by">];
  let Args = [ExprArgument<"Arg">];
  let LateParsed = 1;
  let TemplateDependent = 1;
}

def PtGuardedBy : InheritableAttr {
  let Spellings = [GNU<"pt_guarded_by">];
  let Args = [ExprArgument<"Arg">];
  let LateParsed = 1;
  let TemplateDependent = 1;
}

def AcquiredAfter : InheritableAttr {
  let Spellings = [GNU<"acquired_after">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
}

def AcquiredBefore : InheritableAttr {
  let Spellings = [GNU<"acquired_before">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
}

def ExclusiveLockFunction : InheritableAttr {
  let Spellings = [GNU<"exclusive_lock_function">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
}

def SharedLockFunction : InheritableAttr {
  let Spellings = [GNU<"shared_lock_function">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
}

// The first argument is an integer or boolean value specifying the return value
// of a successful lock acquisition.
def ExclusiveTrylockFunction : InheritableAttr {
  let Spellings = [GNU<"exclusive_trylock_function">];
  let Args = [ExprArgument<"SuccessValue">, VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
}

// The first argument is an integer or boolean value specifying the return value
// of a successful lock acquisition.
def SharedTrylockFunction : InheritableAttr {
  let Spellings = [GNU<"shared_trylock_function">];
  let Args = [ExprArgument<"SuccessValue">, VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
}

def UnlockFunction : InheritableAttr {
  let Spellings = [GNU<"unlock_function">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
}

def LockReturned : InheritableAttr {
  let Spellings = [GNU<"lock_returned">];
  let Args = [ExprArgument<"Arg">];
  let LateParsed = 1;
  let TemplateDependent = 1;
}

def LocksExcluded : InheritableAttr {
  let Spellings = [GNU<"locks_excluded">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
}

def ExclusiveLocksRequired : InheritableAttr {
  let Spellings = [GNU<"exclusive_locks_required">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
}

def SharedLocksRequired : InheritableAttr {
  let Spellings = [GNU<"shared_locks_required">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
}

// Type safety attributes for `void *' pointers and type tags.

def ArgumentWithTypeTag : InheritableAttr {
  let Spellings = [GNU<"argument_with_type_tag">,
                   GNU<"pointer_with_type_tag">];
  let Args = [IdentifierArgument<"ArgumentKind">,
              UnsignedArgument<"ArgumentIdx">,
              UnsignedArgument<"TypeTagIdx">,
              BoolArgument<"IsPointer">];
  let Subjects = [Function];
}

def TypeTagForDatatype : InheritableAttr {
  let Spellings = [GNU<"type_tag_for_datatype">];
  let Args = [IdentifierArgument<"ArgumentKind">,
              TypeArgument<"MatchingCType">,
              BoolArgument<"LayoutCompatible">,
              BoolArgument<"MustBeNull">];
  let Subjects = [Var];
}

// Microsoft-related attributes

def MsStruct : InheritableAttr {
  let Spellings = [Declspec<"ms_struct">];
}

def DLLExport : InheritableAttr {
  let Spellings = [Declspec<"dllexport">];
}

def DLLImport : InheritableAttr {
  let Spellings = [Declspec<"dllimport">];
}

def ForceInline : InheritableAttr {
  let Spellings = [Declspec<"__forceinline">];
}

def Win64 : InheritableAttr {
  let Spellings = [Declspec<"w64">];
}

def Ptr32 : InheritableAttr {
  let Spellings = [Declspec<"__ptr32">];
}

def Ptr64 : InheritableAttr {
  let Spellings = [Declspec<"__ptr64">];
}

def SingleInheritance : InheritableAttr {
  let Spellings = [Declspec<"__single_inheritance">];
}

def MultipleInheritance : InheritableAttr {
  let Spellings = [Declspec<"__multiple_inheritance">];
}

def VirtualInheritance : InheritableAttr {
  let Spellings = [Declspec<"__virtual_inheritance">];
}
