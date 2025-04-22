#ifdef GET_ATTR_NAMES
#undef GET_ATTR_NAMES
#ifndef ATTRIBUTE_ALL
#define ATTRIBUTE_ALL(FIRST, SECOND)
#endif

#ifndef ATTRIBUTE_ENUM
#define ATTRIBUTE_ENUM(FIRST, SECOND) ATTRIBUTE_ALL(FIRST, SECOND)
#endif

ATTRIBUTE_ENUM(AllocAlign,allocalign)
ATTRIBUTE_ENUM(AllocatedPointer,allocptr)
ATTRIBUTE_ENUM(AlwaysInline,alwaysinline)
ATTRIBUTE_ENUM(Builtin,builtin)
ATTRIBUTE_ENUM(Cold,cold)
ATTRIBUTE_ENUM(Convergent,convergent)
ATTRIBUTE_ENUM(CoroDestroyOnlyWhenComplete,coro_only_destroy_when_complete)
ATTRIBUTE_ENUM(CoroElideSafe,coro_elide_safe)
ATTRIBUTE_ENUM(DeadOnUnwind,dead_on_unwind)
ATTRIBUTE_ENUM(DisableSanitizerInstrumentation,disable_sanitizer_instrumentation)
ATTRIBUTE_ENUM(FnRetThunkExtern,fn_ret_thunk_extern)
ATTRIBUTE_ENUM(Hot,hot)
ATTRIBUTE_ENUM(HybridPatchable,hybrid_patchable)
ATTRIBUTE_ENUM(ImmArg,immarg)
ATTRIBUTE_ENUM(InReg,inreg)
ATTRIBUTE_ENUM(InlineHint,inlinehint)
ATTRIBUTE_ENUM(JumpTable,jumptable)
ATTRIBUTE_ENUM(MinSize,minsize)
ATTRIBUTE_ENUM(MustProgress,mustprogress)
ATTRIBUTE_ENUM(Naked,naked)
ATTRIBUTE_ENUM(Nest,nest)
ATTRIBUTE_ENUM(NoAlias,noalias)
ATTRIBUTE_ENUM(NoBuiltin,nobuiltin)
ATTRIBUTE_ENUM(NoCallback,nocallback)
ATTRIBUTE_ENUM(NoCfCheck,nocf_check)
ATTRIBUTE_ENUM(NoDivergenceSource,nodivergencesource)
ATTRIBUTE_ENUM(NoDuplicate,noduplicate)
ATTRIBUTE_ENUM(NoExt,noext)
ATTRIBUTE_ENUM(NoFree,nofree)
ATTRIBUTE_ENUM(NoImplicitFloat,noimplicitfloat)
ATTRIBUTE_ENUM(NoInline,noinline)
ATTRIBUTE_ENUM(NoMerge,nomerge)
ATTRIBUTE_ENUM(NoProfile,noprofile)
ATTRIBUTE_ENUM(NoRecurse,norecurse)
ATTRIBUTE_ENUM(NoRedZone,noredzone)
ATTRIBUTE_ENUM(NoReturn,noreturn)
ATTRIBUTE_ENUM(NoSanitizeBounds,nosanitize_bounds)
ATTRIBUTE_ENUM(NoSanitizeCoverage,nosanitize_coverage)
ATTRIBUTE_ENUM(NoSync,nosync)
ATTRIBUTE_ENUM(NoUndef,noundef)
ATTRIBUTE_ENUM(NoUnwind,nounwind)
ATTRIBUTE_ENUM(NonLazyBind,nonlazybind)
ATTRIBUTE_ENUM(NonNull,nonnull)
ATTRIBUTE_ENUM(NullPointerIsValid,null_pointer_is_valid)
ATTRIBUTE_ENUM(OptForFuzzing,optforfuzzing)
ATTRIBUTE_ENUM(OptimizeForDebugging,optdebug)
ATTRIBUTE_ENUM(OptimizeForSize,optsize)
ATTRIBUTE_ENUM(OptimizeNone,optnone)
ATTRIBUTE_ENUM(PresplitCoroutine,presplitcoroutine)
ATTRIBUTE_ENUM(ReadNone,readnone)
ATTRIBUTE_ENUM(ReadOnly,readonly)
ATTRIBUTE_ENUM(Returned,returned)
ATTRIBUTE_ENUM(ReturnsTwice,returns_twice)
ATTRIBUTE_ENUM(SExt,signext)
ATTRIBUTE_ENUM(SafeStack,safestack)
ATTRIBUTE_ENUM(SanitizeAddress,sanitize_address)
ATTRIBUTE_ENUM(SanitizeHWAddress,sanitize_hwaddress)
ATTRIBUTE_ENUM(SanitizeMemTag,sanitize_memtag)
ATTRIBUTE_ENUM(SanitizeMemory,sanitize_memory)
ATTRIBUTE_ENUM(SanitizeNumericalStability,sanitize_numerical_stability)
ATTRIBUTE_ENUM(SanitizeRealtime,sanitize_realtime)
ATTRIBUTE_ENUM(SanitizeRealtimeBlocking,sanitize_realtime_blocking)
ATTRIBUTE_ENUM(SanitizeThread,sanitize_thread)
ATTRIBUTE_ENUM(SanitizeType,sanitize_type)
ATTRIBUTE_ENUM(ShadowCallStack,shadowcallstack)
ATTRIBUTE_ENUM(SkipProfile,skipprofile)
ATTRIBUTE_ENUM(Speculatable,speculatable)
ATTRIBUTE_ENUM(SpeculativeLoadHardening,speculative_load_hardening)
ATTRIBUTE_ENUM(StackProtect,ssp)
ATTRIBUTE_ENUM(StackProtectReq,sspreq)
ATTRIBUTE_ENUM(StackProtectStrong,sspstrong)
ATTRIBUTE_ENUM(StrictFP,strictfp)
ATTRIBUTE_ENUM(SwiftAsync,swiftasync)
ATTRIBUTE_ENUM(SwiftError,swifterror)
ATTRIBUTE_ENUM(SwiftSelf,swiftself)
ATTRIBUTE_ENUM(WillReturn,willreturn)
ATTRIBUTE_ENUM(Writable,writable)
ATTRIBUTE_ENUM(WriteOnly,writeonly)
ATTRIBUTE_ENUM(ZExt,zeroext)
ATTRIBUTE_ENUM(ByRef,byref)
ATTRIBUTE_ENUM(ByVal,byval)
ATTRIBUTE_ENUM(ElementType,elementtype)
ATTRIBUTE_ENUM(InAlloca,inalloca)
ATTRIBUTE_ENUM(Preallocated,preallocated)
ATTRIBUTE_ENUM(StructRet,sret)
ATTRIBUTE_ENUM(Alignment,align)
ATTRIBUTE_ENUM(AllocKind,allockind)
ATTRIBUTE_ENUM(AllocSize,allocsize)
ATTRIBUTE_ENUM(Captures,captures)
ATTRIBUTE_ENUM(Dereferenceable,dereferenceable)
ATTRIBUTE_ENUM(DereferenceableOrNull,dereferenceable_or_null)
ATTRIBUTE_ENUM(Memory,memory)
ATTRIBUTE_ENUM(NoFPClass,nofpclass)
ATTRIBUTE_ENUM(StackAlignment,alignstack)
ATTRIBUTE_ENUM(UWTable,uwtable)
ATTRIBUTE_ENUM(VScaleRange,vscale_range)
ATTRIBUTE_ENUM(Range,range)
ATTRIBUTE_ENUM(Initializes,initializes)
#undef ATTRIBUTE_ENUM

#ifndef ATTRIBUTE_STRBOOL
#define ATTRIBUTE_STRBOOL(FIRST, SECOND) ATTRIBUTE_ALL(FIRST, SECOND)
#endif

ATTRIBUTE_STRBOOL(ApproxFuncFPMath,approx-func-fp-math)
ATTRIBUTE_STRBOOL(LessPreciseFPMAD,less-precise-fpmad)
ATTRIBUTE_STRBOOL(NoInfsFPMath,no-infs-fp-math)
ATTRIBUTE_STRBOOL(NoInlineLineTables,no-inline-line-tables)
ATTRIBUTE_STRBOOL(NoJumpTables,no-jump-tables)
ATTRIBUTE_STRBOOL(NoNansFPMath,no-nans-fp-math)
ATTRIBUTE_STRBOOL(NoSignedZerosFPMath,no-signed-zeros-fp-math)
ATTRIBUTE_STRBOOL(ProfileSampleAccurate,profile-sample-accurate)
ATTRIBUTE_STRBOOL(UnsafeFPMath,unsafe-fp-math)
ATTRIBUTE_STRBOOL(UseSampleProfile,use-sample-profile)
#undef ATTRIBUTE_STRBOOL

#ifndef ATTRIBUTE_COMPLEXSTR
#define ATTRIBUTE_COMPLEXSTR(FIRST, SECOND) ATTRIBUTE_ALL(FIRST, SECOND)
#endif

ATTRIBUTE_COMPLEXSTR(DenormalFPMath,denormal-fp-math)
ATTRIBUTE_COMPLEXSTR(DenormalFPMathF32,denormal-fp-math-f32)
#undef ATTRIBUTE_COMPLEXSTR

#undef ATTRIBUTE_ALL
#endif

#ifdef GET_ATTR_ENUM
#undef GET_ATTR_ENUM
FirstEnumAttr = 1,
AllocAlign = 1,
AllocatedPointer = 2,
AlwaysInline = 3,
Builtin = 4,
Cold = 5,
Convergent = 6,
CoroDestroyOnlyWhenComplete = 7,
CoroElideSafe = 8,
DeadOnUnwind = 9,
DisableSanitizerInstrumentation = 10,
FnRetThunkExtern = 11,
Hot = 12,
HybridPatchable = 13,
ImmArg = 14,
InReg = 15,
InlineHint = 16,
JumpTable = 17,
MinSize = 18,
MustProgress = 19,
Naked = 20,
Nest = 21,
NoAlias = 22,
NoBuiltin = 23,
NoCallback = 24,
NoCfCheck = 25,
NoDivergenceSource = 26,
NoDuplicate = 27,
NoExt = 28,
NoFree = 29,
NoImplicitFloat = 30,
NoInline = 31,
NoMerge = 32,
NoProfile = 33,
NoRecurse = 34,
NoRedZone = 35,
NoReturn = 36,
NoSanitizeBounds = 37,
NoSanitizeCoverage = 38,
NoSync = 39,
NoUndef = 40,
NoUnwind = 41,
NonLazyBind = 42,
NonNull = 43,
NullPointerIsValid = 44,
OptForFuzzing = 45,
OptimizeForDebugging = 46,
OptimizeForSize = 47,
OptimizeNone = 48,
PresplitCoroutine = 49,
ReadNone = 50,
ReadOnly = 51,
Returned = 52,
ReturnsTwice = 53,
SExt = 54,
SafeStack = 55,
SanitizeAddress = 56,
SanitizeHWAddress = 57,
SanitizeMemTag = 58,
SanitizeMemory = 59,
SanitizeNumericalStability = 60,
SanitizeRealtime = 61,
SanitizeRealtimeBlocking = 62,
SanitizeThread = 63,
SanitizeType = 64,
ShadowCallStack = 65,
SkipProfile = 66,
Speculatable = 67,
SpeculativeLoadHardening = 68,
StackProtect = 69,
StackProtectReq = 70,
StackProtectStrong = 71,
StrictFP = 72,
SwiftAsync = 73,
SwiftError = 74,
SwiftSelf = 75,
WillReturn = 76,
Writable = 77,
WriteOnly = 78,
ZExt = 79,
LastEnumAttr = 79,
FirstTypeAttr = 80,
ByRef = 80,
ByVal = 81,
ElementType = 82,
InAlloca = 83,
Preallocated = 84,
StructRet = 85,
LastTypeAttr = 85,
FirstIntAttr = 86,
Alignment = 86,
AllocKind = 87,
AllocSize = 88,
Captures = 89,
Dereferenceable = 90,
DereferenceableOrNull = 91,
Memory = 92,
NoFPClass = 93,
StackAlignment = 94,
UWTable = 95,
VScaleRange = 96,
LastIntAttr = 96,
FirstConstantRangeAttr = 97,
Range = 97,
LastConstantRangeAttr = 97,
FirstConstantRangeListAttr = 98,
Initializes = 98,
LastConstantRangeListAttr = 98,
#endif

#ifdef GET_ATTR_COMPAT_FUNC
#undef GET_ATTR_COMPAT_FUNC
static inline bool hasCompatibleFnAttrs(const Function &Caller,
                                        const Function &Callee) {
  bool Ret = true;

  Ret &= isEqual<SanitizeAddressAttr>(Caller, Callee);
  Ret &= isEqual<SanitizeThreadAttr>(Caller, Callee);
  Ret &= isEqual<SanitizeTypeAttr>(Caller, Callee);
  Ret &= isEqual<SanitizeMemoryAttr>(Caller, Callee);
  Ret &= isEqual<SanitizeHWAddressAttr>(Caller, Callee);
  Ret &= isEqual<SanitizeMemTagAttr>(Caller, Callee);
  Ret &= isEqual<SanitizeNumericalStabilityAttr>(Caller, Callee);
  Ret &= isEqual<SanitizeRealtimeAttr>(Caller, Callee);
  Ret &= isEqual<SanitizeRealtimeBlockingAttr>(Caller, Callee);
  Ret &= isEqual<SafeStackAttr>(Caller, Callee);
  Ret &= isEqual<ShadowCallStackAttr>(Caller, Callee);
  Ret &= isEqual<UseSampleProfileAttr>(Caller, Callee);
  Ret &= isEqual<NoProfileAttr>(Caller, Callee);
  Ret &= checkDenormMode(Caller, Callee);
  Ret &= checkStrictFP(Caller, Callee);
  Ret &= isEqual(Caller, Callee, "sign-return-address");
  Ret &= isEqual(Caller, Callee, "sign-return-address-key");
  Ret &= isEqual(Caller, Callee, "branch-protection-pauth-lr");

  return Ret;
}

static inline void mergeFnAttrs(Function &Caller,
                                const Function &Callee) {
  setAND<LessPreciseFPMADAttr>(Caller, Callee);
  setAND<NoInfsFPMathAttr>(Caller, Callee);
  setAND<NoNansFPMathAttr>(Caller, Callee);
  setAND<ApproxFuncFPMathAttr>(Caller, Callee);
  setAND<NoSignedZerosFPMathAttr>(Caller, Callee);
  setAND<UnsafeFPMathAttr>(Caller, Callee);
  setOR<NoImplicitFloatAttr>(Caller, Callee);
  setOR<NoJumpTablesAttr>(Caller, Callee);
  setOR<ProfileSampleAccurateAttr>(Caller, Callee);
  setOR<SpeculativeLoadHardeningAttr>(Caller, Callee);
  adjustCallerSSPLevel(Caller, Callee);
  adjustCallerStackProbes(Caller, Callee);
  adjustCallerStackProbeSize(Caller, Callee);
  adjustMinLegalVectorWidth(Caller, Callee);
  adjustNullPointerValidAttr(Caller, Callee);
  setAND<MustProgressAttr>(Caller, Callee);
}

#endif
#ifdef GET_ATTR_PROP_TABLE
#undef GET_ATTR_PROP_TABLE
static const uint8_t AttrPropTable[] = {
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::ParamAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectAnd,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::ParamAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr | AttributeProperty::IntersectCustom,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectCustom,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr | AttributeProperty::IntersectMin,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr | AttributeProperty::IntersectMin,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectCustom,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr | AttributeProperty::IntersectCustom,
0 | AttributeProperty::FnAttr | AttributeProperty::ParamAttr | AttributeProperty::RetAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::FnAttr | AttributeProperty::IntersectPreserve,
0 | AttributeProperty::ParamAttr | AttributeProperty::RetAttr | AttributeProperty::IntersectCustom,
0 | AttributeProperty::ParamAttr | AttributeProperty::IntersectPreserve,
};
#endif
