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

// This is the type of a variable which C++11 allows alignas(...) to appertain
// to.
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
class Keyword<string name> : Spelling<name, "Keyword">;

class Accessor<string name, list<Spelling> spellings> {
  string Name = name;
  list<Spelling> Spellings = spellings;
}

class Attr {
  // The various ways in which an attribute can be spelled in source
  list<Spelling> Spellings;
  // The things to which an attribute can appertain
  list<AttrSubject> Subjects;
  // The arguments allowed on an attribute
  list<Argument> Args = [];
  // Accessors which should be generated for the attribute.
  list<Accessor> Accessors = [];
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

/// An ignored attribute, which we parse but discard with no checking.
class IgnoredAttr : Attr {
  let Ignored = 1;
  let ASTNode = 0;
  let SemaHandler = 0;
}

//
// Attributes begin here
//

def AddressSpace : Attr {
  let Spellings = [GNU<"address_space">];
  let Args = [IntArgument<"AddressSpace">];
  let ASTNode = 0;
}

def Alias : InheritableAttr {
  let Spellings = [GNU<"alias">, CXX11<"gnu", "alias">];
  let Args = [StringArgument<"Aliasee">];
}

def Aligned : InheritableAttr {
  let Spellings = [GNU<"aligned">, Declspec<"align">, CXX11<"gnu", "aligned">,
                   Keyword<"alignas">, Keyword<"_Alignas">];
  let Subjects = [NonBitField, NormalVar, Tag];
  let Args = [AlignedArgument<"Alignment">];
  let Accessors = [Accessor<"isGNU", [GNU<"aligned">, CXX11<"gnu","aligned">]>,
                   Accessor<"isC11", [Keyword<"_Alignas">]>,
                   Accessor<"isAlignas", [Keyword<"alignas">,
                                          Keyword<"_Alignas">]>,
                   Accessor<"isDeclspec",[Declspec<"align">]>];
}

def AlignMac68k : InheritableAttr {
  let Spellings = [];
  let SemaHandler = 0;
}

def AllocSize : Attr {
  let Spellings = [GNU<"alloc_size">, CXX11<"gnu", "alloc_size">];
  let Args = [VariadicUnsignedArgument<"Args">];
}

def AlwaysInline : InheritableAttr {
  let Spellings = [GNU<"always_inline">, CXX11<"gnu", "always_inline">];
}

def TLSModel : InheritableAttr {
  let Spellings = [GNU<"tls_model">, CXX11<"gnu", "tls_model">];
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

def Bounded : IgnoredAttr {
  let Spellings = [GNU<"bounded">];
}

def CarriesDependency : InheritableParamAttr {
  let Spellings = [GNU<"carries_dependency">, CXX11<"","carries_dependency">,
                   CXX11<"std","carries_dependency">];
  let Subjects = [ParmVar, Function];
}

def CDecl : InheritableAttr {
  let Spellings = [GNU<"cdecl">, CXX11<"gnu", "cdecl">, Keyword<"__cdecl">,
                   Keyword<"_cdecl">];
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
  let Spellings = [GNU<"cleanup">, CXX11<"gnu", "cleanup">];
  let Args = [FunctionArgument<"FunctionDecl">];
}

def Cold : InheritableAttr {
  let Spellings = [GNU<"cold">, CXX11<"gnu", "cold">];
}

def Common : InheritableAttr {
  let Spellings = [GNU<"common">, CXX11<"gnu", "common">];
}

def Const : InheritableAttr {
  let Spellings = [GNU<"const">, GNU<"__const">, CXX11<"gnu", "const">];
}

def Constructor : InheritableAttr {
  let Spellings = [GNU<"constructor">, CXX11<"gnu", "constructor">];
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

def C11NoReturn : InheritableAttr {
  let Spellings = [Keyword<"_Noreturn">];
  let Subjects = [Function];
  let SemaHandler = 0;
}

def CXX11NoReturn : InheritableAttr {
  let Spellings = [CXX11<"","noreturn">, CXX11<"std","noreturn">];
  let Subjects = [Function];
}

def OpenCLKernel : Attr {
  let Spellings = [Keyword<"__kernel">, Keyword<"kernel">];
}

def OpenCLImageAccess : Attr {
  let Spellings = [GNU<"opencl_image_access">];
  let Args = [IntArgument<"Access">];
}

def Deprecated : InheritableAttr {
  let Spellings = [GNU<"deprecated">, CXX11<"gnu", "deprecated">];
  let Args = [StringArgument<"Message">];
}

def Destructor : InheritableAttr {
  let Spellings = [GNU<"destructor">, CXX11<"gnu", "destructor">];
  let Args = [IntArgument<"Priority">];
}

def ExtVectorType : Attr {
  let Spellings = [GNU<"ext_vector_type">];
  let Args = [ExprArgument<"NumElements">];
  let ASTNode = 0;
}

def FallThrough : Attr {
  let Spellings = [CXX11<"clang", "fallthrough">];
  let Subjects = [NullStmt];
}

def FastCall : InheritableAttr {
  let Spellings = [GNU<"fastcall">, CXX11<"gnu", "fastcall">,
                   Keyword<"__fastcall">, Keyword<"_fastcall">];
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
  let Spellings = [GNU<"format">, CXX11<"gnu", "format">];
  let Args = [StringArgument<"Type">, IntArgument<"FormatIdx">,
              IntArgument<"FirstArg">];
}

def FormatArg : InheritableAttr {
  let Spellings = [GNU<"format_arg">, CXX11<"gnu", "format_arg">];
  let Args = [IntArgument<"FormatIdx">];
}

def GNUInline : InheritableAttr {
  let Spellings = [GNU<"gnu_inline">, CXX11<"gnu", "gnu_inline">];
}

def Hot : InheritableAttr {
  let Spellings = [GNU<"hot">, CXX11<"gnu", "hot">];
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
  let Spellings = [GNU<"malloc">, CXX11<"gnu", "malloc">];
}

def MaxFieldAlignment : InheritableAttr {
  let Spellings = [];
  let Args = [UnsignedArgument<"Alignment">];
  let SemaHandler = 0;
}

def MayAlias : InheritableAttr {
  let Spellings = [GNU<"may_alias">, CXX11<"gnu", "may_alias">];
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

def Mips16 : InheritableAttr {
  let Spellings = [GNU<"mips16">, CXX11<"gnu", "mips16">];
  let Subjects = [Function];
}

def Mode : Attr {
  let Spellings = [GNU<"mode">, CXX11<"gnu", "mode">];
  let Args = [IdentifierArgument<"Mode">];
  let ASTNode = 0;
}

def Naked : InheritableAttr {
  let Spellings = [GNU<"naked">, CXX11<"gnu", "naked">];
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
  let Spellings = [GNU<"returns_twice">, CXX11<"gnu", "returns_twice">];
}

def NoCommon : InheritableAttr {
  let Spellings = [GNU<"nocommon">, CXX11<"gnu", "nocommon">];
}

def NoDebug : InheritableAttr {
  let Spellings = [GNU<"nodebug">];
}

def NoInline : InheritableAttr {
  let Spellings = [GNU<"noinline">, CXX11<"gnu", "noinline">];
}

def NoMips16 : InheritableAttr {
  let Spellings = [GNU<"nomips16">, CXX11<"gnu", "nomips16">];
  let Subjects = [Function];
}

def NonNull : InheritableAttr {
  let Spellings = [GNU<"nonnull">, CXX11<"gnu", "nonnull">];
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
  let Spellings = [GNU<"noreturn">, CXX11<"gnu", "noreturn">];
  // FIXME: Does GCC allow this on the function instead?
  let Subjects = [Function];
}

def NoInstrumentFunction : InheritableAttr {
  let Spellings = [GNU<"no_instrument_function">,
                   CXX11<"gnu", "no_instrument_function">];
  let Subjects = [Function];
}

def NoThrow : InheritableAttr {
  let Spellings = [GNU<"nothrow">, CXX11<"gnu", "nothrow">];
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
  let Spellings = [GNU<"packed">, CXX11<"gnu", "packed">];
}

def PnaclCall : InheritableAttr {
  let Spellings = [GNU<"pnaclcall">];
}

def IntelOclBicc : InheritableAttr {
  let Spellings = [GNU<"intel_ocl_bicc">];
}

def Pcs : InheritableAttr {
  let Spellings = [GNU<"pcs">, CXX11<"gnu", "pcs">];
  let Args = [EnumArgument<"PCS", "PCSType",
                           ["aapcs", "aapcs-vfp"],
                           ["AAPCS", "AAPCS_VFP"]>];
}

def Pure : InheritableAttr {
  let Spellings = [GNU<"pure">, CXX11<"gnu", "pure">];
}

def Regparm : InheritableAttr {
  let Spellings = [GNU<"regparm">, CXX11<"gnu", "regparm">];
  let Args = [UnsignedArgument<"NumParams">];
}

def ReqdWorkGroupSize : InheritableAttr {
  let Spellings = [GNU<"reqd_work_group_size">];
  let Args = [UnsignedArgument<"XDim">, UnsignedArgument<"YDim">,
              UnsignedArgument<"ZDim">];
}

def Endian : InheritableAttr {
  let Spellings = [GNU<"endian">];
  let Args = [IdentifierArgument<"platform">];
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
  let Spellings = [GNU<"section">, CXX11<"gnu", "section">];
  let Args = [StringArgument<"Name">];
}

def Sentinel : InheritableAttr {
  let Spellings = [GNU<"sentinel">, CXX11<"gnu", "sentinel">];
  let Args = [DefaultIntArgument<"Sentinel", 0>,
              DefaultIntArgument<"NullPos", 0>];
}

def StdCall : InheritableAttr {
  let Spellings = [GNU<"stdcall">, CXX11<"gnu", "stdcall">,
                   Keyword<"__stdcall">, Keyword<"_stdcall">];
}

def ThisCall : InheritableAttr {
  let Spellings = [GNU<"thiscall">, CXX11<"gnu", "thiscall">,
                   Keyword<"__thiscall">, Keyword<"_thiscall">];
}

def Pascal : InheritableAttr {
  let Spellings = [GNU<"pascal">, Keyword<"__pascal">, Keyword<"_pascal">];
}

def TransparentUnion : InheritableAttr {
  let Spellings = [GNU<"transparent_union">, CXX11<"gnu", "transparent_union">];
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
  let Spellings = [GNU<"unused">, CXX11<"gnu", "unused">];
}

def Used : InheritableAttr {
  let Spellings = [GNU<"used">, CXX11<"gnu", "used">];
}

def Uuid : InheritableAttr {
  let Spellings = [GNU<"uuid">];
  let Args = [StringArgument<"Guid">];
  let Subjects = [CXXRecord];
}

def VectorSize : Attr {
  let Spellings = [GNU<"vector_size">, CXX11<"gnu", "vector_size">];
  let Args = [ExprArgument<"NumBytes">];
  let ASTNode = 0;
}

def VecTypeHint : InheritableAttr {
  let Spellings = [GNU<"vec_type_hint">];
  let Args = [TypeArgument<"TypeHint">, SourceLocArgument<"TypeLoc">];
}

def Visibility : InheritableAttr {
  let Clone = 0;
  let Spellings = [GNU<"visibility">, CXX11<"gnu", "visibility">];
  let Args = [EnumArgument<"Visibility", "VisibilityType",
                           ["default", "hidden", "internal", "protected"],
                           ["Default", "Hidden", "Hidden", "Protected"]>];
}

def TypeVisibility : InheritableAttr {
  let Clone = 0;
  let Spellings = [GNU<"type_visibility">, CXX11<"clang", "type_visibility">];
  let Args = [EnumArgument<"Visibility", "VisibilityType",
                           ["default", "hidden", "internal", "protected"],
                           ["Default", "Hidden", "Hidden", "Protected"]>];
}

def VecReturn : InheritableAttr {
  let Spellings = [GNU<"vecreturn">];
  let Subjects = [CXXRecord];
}

def WarnUnusedResult : InheritableAttr {
  let Spellings = [GNU<"warn_unused_result">,
                   CXX11<"clang", "warn_unused_result">,
                   CXX11<"gnu", "warn_unused_result">];
}

def Weak : InheritableAttr {
  let Spellings = [GNU<"weak">, CXX11<"gnu", "weak">];
}

def WeakImport : InheritableAttr {
  let Spellings = [GNU<"weak_import">];
}

def WeakRef : InheritableAttr {
  let Spellings = [GNU<"weakref">, CXX11<"gnu", "weakref">];
}

def X86ForceAlignArgPointer : InheritableAttr {
  let Spellings = [];
}

// Attribute to disable AddressSanitizer (or equivalent) checks.
def NoSanitizeAddress : InheritableAttr {
  let Spellings = [GNU<"no_address_safety_analysis">,
                   GNU<"no_sanitize_address">];
}

// Attribute to disable ThreadSanitizer checks.
def NoSanitizeThread : InheritableAttr {
  let Spellings = [GNU<"no_sanitize_thread">];
}

// Attribute to disable MemorySanitizer checks.
def NoSanitizeMemory : InheritableAttr {
  let Spellings = [GNU<"no_sanitize_memory">];
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

def MsProperty : Attr {
  let Spellings = [Declspec<"property">];
}

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
  let Spellings = [Keyword<"__forceinline">];
}

def SelectAny : InheritableAttr {
  let Spellings = [Declspec<"selectany">];
}

def Win64 : InheritableAttr {
  let Spellings = [Keyword<"__w64">];
}

def Ptr32 : InheritableAttr {
  let Spellings = [Keyword<"__ptr32">];
}

def Ptr64 : InheritableAttr {
  let Spellings = [Keyword<"__ptr64">];
}

class MSInheritanceAttr : InheritableAttr;

def SingleInheritance : MSInheritanceAttr {
  let Spellings = [Keyword<"__single_inheritance">];
}

def MultipleInheritance : MSInheritanceAttr {
  let Spellings = [Keyword<"__multiple_inheritance">];
}

def VirtualInheritance : MSInheritanceAttr {
  let Spellings = [Keyword<"__virtual_inheritance">];
}

// This attribute doesn't have any spellings, but we can apply it implicitly to
// incomplete types that lack any of the other attributes.
def UnspecifiedInheritance : MSInheritanceAttr {
  let Spellings = [];
}

def Unaligned : IgnoredAttr {
  let Spellings = [Keyword<"__unaligned">];
}
