// RUN: clang-tblgen -gen-clang-attr-undocumented-list -I%S/../../include/ %S/../../include/clang/Basic/Attr.td -o - | FileCheck %s

This test serves to prevent us adding new attributes to Clang that are
undocumented. All new attributes should be documented.

The list of attributes below should NEVER grow. It should gradually shrink to 0.

CHECK-LABEL: Undocumented attributes:
CHECK-NEXT:	AcquiredAfter
CHECK-NEXT:	AcquiredBefore
CHECK-NEXT:	Alias
CHECK-NEXT:	Aligned
CHECK-NEXT:	AnalyzerNoReturn
CHECK-NEXT:	ArcWeakrefUnavailable
CHECK-NEXT:	AvailableOnlyInDefaultEvalMethod
CHECK-NEXT:	Blocks
CHECK-NEXT:	CDecl
CHECK-NEXT:	CFAuditedTransfer
CHECK-NEXT:	CFUnknownTransfer
CHECK-NEXT:	CUDAConstant
CHECK-NEXT:	CUDADevice
CHECK-NEXT:	CUDAGlobal
CHECK-NEXT:	CUDAHost
CHECK-NEXT:	CUDALaunchBounds
CHECK-NEXT:	CUDAShared
CHECK-NEXT:	Capability
CHECK-NEXT:	Common
CHECK-NEXT:	Const
CHECK-NEXT:	ConsumableAutoCast
CHECK-NEXT:	ConsumableSetOnRead
CHECK-NEXT:	FormatArg
CHECK-NEXT:	GuardedBy
CHECK-NEXT:	GuardedVar
CHECK-NEXT:	IBAction
CHECK-NEXT:	IBOutlet
CHECK-NEXT:	IBOutletCollection
CHECK-NEXT:	IntelOclBicc
CHECK-NEXT:	LockReturned
CHECK-NEXT:	Lockable
CHECK-NEXT:	LocksExcluded
CHECK-NEXT:	M68kInterrupt
CHECK-NEXT:	MSP430Interrupt
CHECK-NEXT:	MatrixType
CHECK-NEXT:	MayAlias
CHECK-NEXT:	Mips16
CHECK-NEXT:	Mode
CHECK-NEXT:	Naked
CHECK-NEXT:	NeonPolyVectorType
CHECK-NEXT:	NeonVectorType
CHECK-NEXT:	NoCommon
CHECK-NEXT:	NoFieldProtection
CHECK-NEXT:	NoInstrumentFunction
CHECK-NEXT:	NoMips16
CHECK-NEXT:	NoReturn
CHECK-NEXT:	NoThreadSafetyAnalysis
CHECK-NEXT:	ObjCBridge
CHECK-NEXT:	ObjCBridgeMutable
CHECK-NEXT:	ObjCBridgeRelated
CHECK-NEXT:	ObjCDesignatedInitializer
CHECK-NEXT:	ObjCException
CHECK-NEXT:	ObjCExplicitProtocolImpl
CHECK-NEXT:	ObjCGC
CHECK-NEXT:	ObjCIndependentClass
CHECK-NEXT:	ObjCKindOf
CHECK-NEXT:	ObjCNSObject
CHECK-NEXT:	ObjCOwnership
CHECK-NEXT:	ObjCPreciseLifetime
CHECK-NEXT:	ObjCRequiresPropertyDefs
CHECK-NEXT:	ObjCReturnsInnerPointer
CHECK-NEXT:	ObjCRootClass
CHECK-NEXT:	OverflowBehavior
CHECK-NEXT:	Packed
CHECK-NEXT:	Pascal
CHECK-NEXT:	PointerFieldProtection
CHECK-NEXT:	PtGuardedBy
CHECK-NEXT:	PtGuardedVar
CHECK-NEXT:	Pure
CHECK-NEXT:	ReentrantCapability
CHECK-NEXT:	ReqdWorkGroupSize
CHECK-NEXT:	RequiresCapability
CHECK-NEXT:	ReturnsTwice
CHECK-NEXT:	ScopedLockable
CHECK-NEXT:	Unavailable
CHECK-NEXT:	Uuid
CHECK-NEXT:	VTablePointerAuthentication
CHECK-NEXT:	VecReturn
CHECK-NEXT:	VecTypeHint
CHECK-NEXT:	VectorSize
CHECK-NEXT:	Visibility
CHECK-NEXT:	WeakImport
CHECK-NEXT:	WeakRef
CHECK-NEXT:	WorkGroupSizeHint
CHECK-NEXT: Total: 84
