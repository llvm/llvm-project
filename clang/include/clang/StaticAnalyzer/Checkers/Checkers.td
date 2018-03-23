//===--- Checkers.td - Static Analyzer Checkers -===-----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

include "clang/StaticAnalyzer/Checkers/CheckerBase.td"

//===----------------------------------------------------------------------===//
// Packages.
//===----------------------------------------------------------------------===//

// The Alpha package is for checkers that have too many false positives to be
// turned on by default. The hierarchy under Alpha should be organized in the
// hierarchy checkers would have had if they were truly at the top level.
// (For example, a Cocoa-specific checker that is alpha should be in
// alpha.osx.cocoa).
def Alpha : Package<"alpha">;

def Core : Package<"core">;
def CoreBuiltin : Package<"builtin">, InPackage<Core>;
def CoreUninitialized  : Package<"uninitialized">, InPackage<Core>;
def CoreAlpha : Package<"core">, InPackage<Alpha>, Hidden;

// The OptIn package is for checkers that are not alpha and that would normally
// be on by default but where the driver does not have enough information to
// determine when they are applicable. For example, localizability checkers fit
// this criterion because the driver cannot determine whether a project is
// localized or not -- this is best determined at the IDE or build-system level.
//
// The checker hierarchy under OptIn should mirror that in Alpha: checkers
// should be organized as if they were at the top level.
//
// Note: OptIn is *not* intended for checkers that are too noisy to be on by
// default. Such checkers belong in the alpha package.
def OptIn : Package<"optin">;

// In the Portability package reside checkers for finding code that relies on
// implementation-defined behavior. Such checks are wanted for cross-platform
// development, but unwanted for developers who target only a single platform.
def PortabilityOptIn : Package<"portability">, InPackage<OptIn>;

def Nullability : Package<"nullability">;

def Cplusplus : Package<"cplusplus">;
def CplusplusAlpha : Package<"cplusplus">, InPackage<Alpha>, Hidden;
def CplusplusOptIn : Package<"cplusplus">, InPackage<OptIn>;

def Valist : Package<"valist">;

def DeadCode : Package<"deadcode">;
def DeadCodeAlpha : Package<"deadcode">, InPackage<Alpha>, Hidden;

def Performance : Package<"performance">, InPackage<OptIn>;

def Security : Package <"security">;
def InsecureAPI : Package<"insecureAPI">, InPackage<Security>;
def SecurityAlpha : Package<"security">, InPackage<Alpha>, Hidden;
def Taint : Package<"taint">, InPackage<SecurityAlpha>, Hidden;

def Unix : Package<"unix">;
def UnixAlpha : Package<"unix">, InPackage<Alpha>, Hidden;
def CString : Package<"cstring">, InPackage<Unix>, Hidden;
def CStringAlpha : Package<"cstring">, InPackage<UnixAlpha>, Hidden;

def OSX : Package<"osx">;
def OSXAlpha : Package<"osx">, InPackage<Alpha>, Hidden;
def OSXOptIn : Package<"osx">, InPackage<OptIn>;

def Cocoa : Package<"cocoa">, InPackage<OSX>;
def CocoaAlpha : Package<"cocoa">, InPackage<OSXAlpha>, Hidden;
def CocoaOptIn : Package<"cocoa">, InPackage<OSXOptIn>;

def CoreFoundation : Package<"coreFoundation">, InPackage<OSX>;
def Containers : Package<"containers">, InPackage<CoreFoundation>;

def LocalizabilityAlpha : Package<"localizability">, InPackage<CocoaAlpha>;
def LocalizabilityOptIn : Package<"localizability">, InPackage<CocoaOptIn>;

def MPI : Package<"mpi">, InPackage<OptIn>;

def LLVM : Package<"llvm">;

// The APIModeling package is for checkers that model APIs and don't perform
// any diagnostics. These checkers are always turned on; this package is
// intended for API modeling that is not controlled by the target triple.
def APIModeling : Package<"apiModeling">, Hidden;
def GoogleAPIModeling : Package<"google">, InPackage<APIModeling>;

def Debug : Package<"debug">;

def CloneDetectionAlpha : Package<"clone">, InPackage<Alpha>, Hidden;

//===----------------------------------------------------------------------===//
// Core Checkers.
//===----------------------------------------------------------------------===//

let ParentPackage = Core in {

def DereferenceChecker : Checker<"NullDereference">,
  HelpText<"Check for dereferences of null pointers">,
  DescFile<"DereferenceChecker.cpp">;

def CallAndMessageChecker : Checker<"CallAndMessage">,
  HelpText<"Check for logical errors for function calls and Objective-C message expressions (e.g., uninitialized arguments, null function pointers)">,
  DescFile<"CallAndMessageChecker.cpp">;

def NonNullParamChecker : Checker<"NonNullParamChecker">,
  HelpText<"Check for null pointers passed as arguments to a function whose arguments are references or marked with the 'nonnull' attribute">,
  DescFile<"NonNullParamChecker.cpp">;

def VLASizeChecker : Checker<"VLASize">,
  HelpText<"Check for declarations of VLA of undefined or zero size">,
  DescFile<"VLASizeChecker.cpp">;

def DivZeroChecker : Checker<"DivideZero">,
  HelpText<"Check for division by zero">,
  DescFile<"DivZeroChecker.cpp">;

def UndefResultChecker : Checker<"UndefinedBinaryOperatorResult">,
  HelpText<"Check for undefined results of binary operators">,
  DescFile<"UndefResultChecker.cpp">;

def StackAddrEscapeChecker : Checker<"StackAddressEscape">,
  HelpText<"Check that addresses to stack memory do not escape the function">,
  DescFile<"StackAddrEscapeChecker.cpp">;

def DynamicTypePropagation : Checker<"DynamicTypePropagation">,
  HelpText<"Generate dynamic type information">,
  DescFile<"DynamicTypePropagation.cpp">;

def NonnullGlobalConstantsChecker: Checker<"NonnilStringConstants">,
  HelpText<"Assume that const string-like globals are non-null">,
  DescFile<"NonilStringConstantsChecker.cpp">;

} // end "core"

let ParentPackage = CoreAlpha in {

def BoolAssignmentChecker : Checker<"BoolAssignment">,
  HelpText<"Warn about assigning non-{0,1} values to Boolean variables">,
  DescFile<"BoolAssignmentChecker.cpp">;

def CastSizeChecker : Checker<"CastSize">,
  HelpText<"Check when casting a malloc'ed type T, whether the size is a multiple of the size of T">,
  DescFile<"CastSizeChecker.cpp">;

def CastToStructChecker : Checker<"CastToStruct">,
  HelpText<"Check for cast from non-struct pointer to struct pointer">,
  DescFile<"CastToStructChecker.cpp">;

def ConversionChecker : Checker<"Conversion">,
  HelpText<"Loss of sign/precision in implicit conversions">,
  DescFile<"ConversionChecker.cpp">;

def IdenticalExprChecker : Checker<"IdenticalExpr">,
  HelpText<"Warn about unintended use of identical expressions in operators">,
  DescFile<"IdenticalExprChecker.cpp">;

def FixedAddressChecker : Checker<"FixedAddr">,
  HelpText<"Check for assignment of a fixed address to a pointer">,
  DescFile<"FixedAddressChecker.cpp">;

def PointerArithChecker : Checker<"PointerArithm">,
  HelpText<"Check for pointer arithmetic on locations other than array elements">,
  DescFile<"PointerArithChecker">;

def PointerSubChecker : Checker<"PointerSub">,
  HelpText<"Check for pointer subtractions on two pointers pointing to different memory chunks">,
  DescFile<"PointerSubChecker">;

def SizeofPointerChecker : Checker<"SizeofPtr">,
  HelpText<"Warn about unintended use of sizeof() on pointer expressions">,
  DescFile<"CheckSizeofPointer.cpp">;

def CallAndMessageUnInitRefArg : Checker<"CallAndMessageUnInitRefArg">,
  HelpText<"Check for logical errors for function calls and Objective-C message expressions (e.g., uninitialized arguments, null function pointers, and pointer to undefined variables)">,
  DescFile<"CallAndMessageChecker.cpp">;

def TestAfterDivZeroChecker : Checker<"TestAfterDivZero">,
  HelpText<"Check for division by variable that is later compared against 0. Either the comparison is useless or there is division by zero.">,
  DescFile<"TestAfterDivZeroChecker.cpp">;

def DynamicTypeChecker : Checker<"DynamicTypeChecker">,
  HelpText<"Check for cases where the dynamic and the static type of an object are unrelated.">,
  DescFile<"DynamicTypeChecker.cpp">;

def StackAddrAsyncEscapeChecker : Checker<"StackAddressAsyncEscape">,
  HelpText<"Check that addresses to stack memory do not escape the function">,
  DescFile<"StackAddrEscapeChecker.cpp">;

} // end "alpha.core"

let ParentPackage = Nullability in {

def NullPassedToNonnullChecker : Checker<"NullPassedToNonnull">,
  HelpText<"Warns when a null pointer is passed to a pointer which has a _Nonnull type.">,
  DescFile<"NullabilityChecker.cpp">;

def NullReturnedFromNonnullChecker : Checker<"NullReturnedFromNonnull">,
  HelpText<"Warns when a null pointer is returned from a function that has _Nonnull return type.">,
  DescFile<"NullabilityChecker.cpp">;

def NullableDereferencedChecker : Checker<"NullableDereferenced">,
  HelpText<"Warns when a nullable pointer is dereferenced.">,
  DescFile<"NullabilityChecker.cpp">;

def NullablePassedToNonnullChecker : Checker<"NullablePassedToNonnull">,
  HelpText<"Warns when a nullable pointer is passed to a pointer which has a _Nonnull type.">,
  DescFile<"NullabilityChecker.cpp">;

def NullableReturnedFromNonnullChecker : Checker<"NullableReturnedFromNonnull">,
  HelpText<"Warns when a nullable pointer is returned from a function that has _Nonnull return type.">,
  DescFile<"NullabilityChecker.cpp">;

} // end "nullability"

//===----------------------------------------------------------------------===//
// Evaluate "builtin" functions.
//===----------------------------------------------------------------------===//

let ParentPackage = CoreBuiltin in {

def NoReturnFunctionChecker : Checker<"NoReturnFunctions">,
  HelpText<"Evaluate \"panic\" functions that are known to not return to the caller">,
  DescFile<"NoReturnFunctionChecker.cpp">;

def BuiltinFunctionChecker : Checker<"BuiltinFunctions">,
  HelpText<"Evaluate compiler builtin functions (e.g., alloca())">,
  DescFile<"BuiltinFunctionChecker.cpp">;

} // end "core.builtin"

//===----------------------------------------------------------------------===//
// Uninitialized values checkers.
//===----------------------------------------------------------------------===//

let ParentPackage = CoreUninitialized in {

def UndefinedArraySubscriptChecker : Checker<"ArraySubscript">,
  HelpText<"Check for uninitialized values used as array subscripts">,
  DescFile<"UndefinedArraySubscriptChecker.cpp">;

def UndefinedAssignmentChecker : Checker<"Assign">,
  HelpText<"Check for assigning uninitialized values">,
  DescFile<"UndefinedAssignmentChecker.cpp">;

def UndefBranchChecker : Checker<"Branch">,
  HelpText<"Check for uninitialized values used as branch conditions">,
  DescFile<"UndefBranchChecker.cpp">;

def UndefCapturedBlockVarChecker : Checker<"CapturedBlockVariable">,
  HelpText<"Check for blocks that capture uninitialized values">,
  DescFile<"UndefCapturedBlockVarChecker.cpp">;

def ReturnUndefChecker : Checker<"UndefReturn">,
  HelpText<"Check for uninitialized values being returned to the caller">,
  DescFile<"ReturnUndefChecker.cpp">;

} // end "core.uninitialized"

//===----------------------------------------------------------------------===//
// C++ checkers.
//===----------------------------------------------------------------------===//

let ParentPackage = Cplusplus in {

def NewDeleteChecker : Checker<"NewDelete">,
  HelpText<"Check for double-free and use-after-free problems. Traces memory managed by new/delete.">,
  DescFile<"MallocChecker.cpp">;

def NewDeleteLeaksChecker : Checker<"NewDeleteLeaks">,
  HelpText<"Check for memory leaks. Traces memory managed by new/delete.">,
  DescFile<"MallocChecker.cpp">;

def CXXSelfAssignmentChecker : Checker<"SelfAssignment">,
  HelpText<"Checks C++ copy and move assignment operators for self assignment">,
  DescFile<"CXXSelfAssignmentChecker.cpp">;

} // end: "cplusplus"

let ParentPackage = CplusplusOptIn in {

def VirtualCallChecker : Checker<"VirtualCall">,
  HelpText<"Check virtual function calls during construction or destruction">,
  DescFile<"VirtualCallChecker.cpp">;

} // end: "optin.cplusplus"

let ParentPackage = CplusplusAlpha in {

def DeleteWithNonVirtualDtorChecker : Checker<"DeleteWithNonVirtualDtor">,
  HelpText<"Reports destructions of polymorphic objects with a non-virtual "
           "destructor in their base class">,
  DescFile<"DeleteWithNonVirtualDtorChecker.cpp">;

def IteratorRangeChecker : Checker<"IteratorRange">,
  HelpText<"Check for iterators used outside their valid ranges">,
  DescFile<"IteratorChecker.cpp">;

def MisusedMovedObjectChecker: Checker<"MisusedMovedObject">,
     HelpText<"Method calls on a moved-from object and copying a moved-from "
              "object will be reported">,
     DescFile<"MisusedMovedObjectChecker.cpp">;

} // end: "alpha.cplusplus"


//===----------------------------------------------------------------------===//
// Valist checkers.
//===----------------------------------------------------------------------===//

let ParentPackage = Valist in {

def UninitializedChecker : Checker<"Uninitialized">,
  HelpText<"Check for usages of uninitialized (or already released) va_lists.">,
  DescFile<"ValistChecker.cpp">;

def UnterminatedChecker : Checker<"Unterminated">,
  HelpText<"Check for va_lists which are not released by a va_end call.">,
  DescFile<"ValistChecker.cpp">;

def CopyToSelfChecker : Checker<"CopyToSelf">,
  HelpText<"Check for va_lists which are copied onto itself.">,
  DescFile<"ValistChecker.cpp">;

} // end : "valist"

//===----------------------------------------------------------------------===//
// Deadcode checkers.
//===----------------------------------------------------------------------===//

let ParentPackage = DeadCode in {

def DeadStoresChecker : Checker<"DeadStores">,
  HelpText<"Check for values stored to variables that are never read afterwards">,
  DescFile<"DeadStoresChecker.cpp">;
} // end DeadCode

let ParentPackage = DeadCodeAlpha in {

def UnreachableCodeChecker : Checker<"UnreachableCode">,
  HelpText<"Check unreachable code">,
  DescFile<"UnreachableCodeChecker.cpp">;

} // end "alpha.deadcode"

//===----------------------------------------------------------------------===//
// Performance checkers.
//===----------------------------------------------------------------------===//

let ParentPackage = Performance in {

def PaddingChecker : Checker<"Padding">,
  HelpText<"Check for excessively padded structs.">,
  DescFile<"PaddingChecker.cpp">;

} // end: "padding"

//===----------------------------------------------------------------------===//
// Security checkers.
//===----------------------------------------------------------------------===//

let ParentPackage = InsecureAPI in {
  def gets : Checker<"gets">,
    HelpText<"Warn on uses of the 'gets' function">,
    DescFile<"CheckSecuritySyntaxOnly.cpp">;
  def getpw : Checker<"getpw">,
    HelpText<"Warn on uses of the 'getpw' function">,
    DescFile<"CheckSecuritySyntaxOnly.cpp">;
  def mktemp : Checker<"mktemp">,
    HelpText<"Warn on uses of the 'mktemp' function">,
    DescFile<"CheckSecuritySyntaxOnly.cpp">;
  def mkstemp : Checker<"mkstemp">,
    HelpText<"Warn when 'mkstemp' is passed fewer than 6 X's in the format string">,
    DescFile<"CheckSecuritySyntaxOnly.cpp">;
  def rand : Checker<"rand">,
    HelpText<"Warn on uses of the 'rand', 'random', and related functions">,
    DescFile<"CheckSecuritySyntaxOnly.cpp">;
  def strcpy : Checker<"strcpy">,
    HelpText<"Warn on uses of the 'strcpy' and 'strcat' functions">,
    DescFile<"CheckSecuritySyntaxOnly.cpp">;
  def vfork : Checker<"vfork">,
    HelpText<"Warn on uses of the 'vfork' function">,
    DescFile<"CheckSecuritySyntaxOnly.cpp">;
  def UncheckedReturn : Checker<"UncheckedReturn">,
    HelpText<"Warn on uses of functions whose return values must be always checked">,
    DescFile<"CheckSecuritySyntaxOnly.cpp">;
}
let ParentPackage = Security in {
  def FloatLoopCounter : Checker<"FloatLoopCounter">,
    HelpText<"Warn on using a floating point value as a loop counter (CERT: FLP30-C, FLP30-CPP)">,
    DescFile<"CheckSecuritySyntaxOnly.cpp">;
}

let ParentPackage = SecurityAlpha in {

def ArrayBoundChecker : Checker<"ArrayBound">,
  HelpText<"Warn about buffer overflows (older checker)">,
  DescFile<"ArrayBoundChecker.cpp">;

def ArrayBoundCheckerV2 : Checker<"ArrayBoundV2">,
  HelpText<"Warn about buffer overflows (newer checker)">,
  DescFile<"ArrayBoundCheckerV2.cpp">;

def ReturnPointerRangeChecker : Checker<"ReturnPtrRange">,
  HelpText<"Check for an out-of-bound pointer being returned to callers">,
  DescFile<"ReturnPointerRangeChecker.cpp">;

def MallocOverflowSecurityChecker : Checker<"MallocOverflow">,
  HelpText<"Check for overflows in the arguments to malloc()">,
  DescFile<"MallocOverflowSecurityChecker.cpp">;

// Operating systems specific PROT_READ/PROT_WRITE values is not implemented,
// the defaults are correct for several common operating systems though, 
// but may need to be overridden via the related analyzer-config flags.
def MmapWriteExecChecker : Checker<"MmapWriteExec">,
  HelpText<"Warn on mmap() calls that are both writable and executable">,
  DescFile<"MmapWriteExecChecker.cpp">;

} // end "alpha.security"

//===----------------------------------------------------------------------===//
// Taint checkers.
//===----------------------------------------------------------------------===//

let ParentPackage = Taint in {

def GenericTaintChecker : Checker<"TaintPropagation">,
  HelpText<"Generate taint information used by other checkers">,
  DescFile<"GenericTaintChecker.cpp">;

} // end "alpha.security.taint"

//===----------------------------------------------------------------------===//
// Unix API checkers.
//===----------------------------------------------------------------------===//

let ParentPackage = Unix in {

def UnixAPIMisuseChecker : Checker<"API">,
  HelpText<"Check calls to various UNIX/Posix functions">,
  DescFile<"UnixAPIChecker.cpp">;

def MallocChecker: Checker<"Malloc">,
  HelpText<"Check for memory leaks, double free, and use-after-free problems. Traces memory managed by malloc()/free().">,
  DescFile<"MallocChecker.cpp">;

def MallocSizeofChecker : Checker<"MallocSizeof">,
  HelpText<"Check for dubious malloc arguments involving sizeof">,
  DescFile<"MallocSizeofChecker.cpp">;

def MismatchedDeallocatorChecker : Checker<"MismatchedDeallocator">,
  HelpText<"Check for mismatched deallocators.">,
  DescFile<"MallocChecker.cpp">;

def VforkChecker : Checker<"Vfork">,
  HelpText<"Check for proper usage of vfork">,
  DescFile<"VforkChecker.cpp">;

def StdCLibraryFunctionsChecker : Checker<"StdCLibraryFunctions">,
  HelpText<"Improve modeling of the C standard library functions">,
  DescFile<"StdLibraryFunctionsChecker.cpp">;

} // end "unix"

let ParentPackage = UnixAlpha in {

def ChrootChecker : Checker<"Chroot">,
  HelpText<"Check improper use of chroot">,
  DescFile<"ChrootChecker.cpp">;

def PthreadLockChecker : Checker<"PthreadLock">,
  HelpText<"Simple lock -> unlock checker">,
  DescFile<"PthreadLockChecker.cpp">;

def StreamChecker : Checker<"Stream">,
  HelpText<"Check stream handling functions">,
  DescFile<"StreamChecker.cpp">;

def SimpleStreamChecker : Checker<"SimpleStream">,
  HelpText<"Check for misuses of stream APIs">,
  DescFile<"SimpleStreamChecker.cpp">;

def BlockInCriticalSectionChecker : Checker<"BlockInCriticalSection">,
  HelpText<"Check for calls to blocking functions inside a critical section">,
  DescFile<"BlockInCriticalSectionChecker.cpp">;

} // end "alpha.unix"

let ParentPackage = CString in {

def CStringNullArg : Checker<"NullArg">,
  HelpText<"Check for null pointers being passed as arguments to C string functions">,
  DescFile<"CStringChecker.cpp">;

def CStringSyntaxChecker : Checker<"BadSizeArg">,
  HelpText<"Check the size argument passed into C string functions for common erroneous patterns">,
  DescFile<"CStringSyntaxChecker.cpp">;
}

let ParentPackage = CStringAlpha in {

def CStringOutOfBounds : Checker<"OutOfBounds">,
  HelpText<"Check for out-of-bounds access in string functions">,
  DescFile<"CStringChecker.cpp">;

def CStringBufferOverlap : Checker<"BufferOverlap">,
  HelpText<"Checks for overlap in two buffer arguments">,
  DescFile<"CStringChecker.cpp">;

def CStringNotNullTerm : Checker<"NotNullTerminated">,
  HelpText<"Check for arguments which are not null-terminating strings">,
  DescFile<"CStringChecker.cpp">;
}

//===----------------------------------------------------------------------===//
// Mac OS X, Cocoa, and Core Foundation checkers.
//===----------------------------------------------------------------------===//

let ParentPackage = OSX in {

def NumberObjectConversionChecker : Checker<"NumberObjectConversion">,
  HelpText<"Check for erroneous conversions of objects representing numbers into numbers">,
  DescFile<"NumberObjectConversionChecker.cpp">;

def MacOSXAPIChecker : Checker<"API">,
  HelpText<"Check for proper uses of various Apple APIs">,
  DescFile<"MacOSXAPIChecker.cpp">;

def MacOSKeychainAPIChecker : Checker<"SecKeychainAPI">,
  HelpText<"Check for proper uses of Secure Keychain APIs">,
  DescFile<"MacOSKeychainAPIChecker.cpp">;

def ObjCPropertyChecker : Checker<"ObjCProperty">,
  HelpText<"Check for proper uses of Objective-C properties">,
  DescFile<"ObjCPropertyChecker.cpp">;

} // end "osx"

let ParentPackage = Cocoa in {

def ObjCAtSyncChecker : Checker<"AtSync">,
  HelpText<"Check for nil pointers used as mutexes for @synchronized">,
  DescFile<"ObjCAtSyncChecker.cpp">;

def NilArgChecker : Checker<"NilArg">,
  HelpText<"Check for prohibited nil arguments to ObjC method calls">,
  DescFile<"BasicObjCFoundationChecks.cpp">;

def ClassReleaseChecker : Checker<"ClassRelease">,
  HelpText<"Check for sending 'retain', 'release', or 'autorelease' directly to a Class">,
  DescFile<"BasicObjCFoundationChecks.cpp">;

def VariadicMethodTypeChecker : Checker<"VariadicMethodTypes">,
  HelpText<"Check for passing non-Objective-C types to variadic collection "
           "initialization methods that expect only Objective-C types">,
  DescFile<"BasicObjCFoundationChecks.cpp">;

def NSAutoreleasePoolChecker : Checker<"NSAutoreleasePool">,
  HelpText<"Warn for suboptimal uses of NSAutoreleasePool in Objective-C GC mode">,
  DescFile<"NSAutoreleasePoolChecker.cpp">;

def ObjCMethSigsChecker : Checker<"IncompatibleMethodTypes">,
  HelpText<"Warn about Objective-C method signatures with type incompatibilities">,
  DescFile<"CheckObjCInstMethSignature.cpp">;

def ObjCUnusedIvarsChecker : Checker<"UnusedIvars">,
  HelpText<"Warn about private ivars that are never used">,
  DescFile<"ObjCUnusedIVarsChecker.cpp">;

def ObjCSelfInitChecker : Checker<"SelfInit">,
  HelpText<"Check that 'self' is properly initialized inside an initializer method">,
  DescFile<"ObjCSelfInitChecker.cpp">;

def ObjCLoopChecker : Checker<"Loops">,
  HelpText<"Improved modeling of loops using Cocoa collection types">,
  DescFile<"BasicObjCFoundationChecks.cpp">;

def ObjCNonNilReturnValueChecker : Checker<"NonNilReturnValue">,
  HelpText<"Model the APIs that are guaranteed to return a non-nil value">,
  DescFile<"BasicObjCFoundationChecks.cpp">;

def ObjCSuperCallChecker : Checker<"MissingSuperCall">,
  HelpText<"Warn about Objective-C methods that lack a necessary call to super">,
  DescFile<"ObjCMissingSuperCallChecker.cpp">;

def NSErrorChecker : Checker<"NSError">,
  HelpText<"Check usage of NSError** parameters">,
  DescFile<"NSErrorChecker.cpp">;

def RetainCountChecker : Checker<"RetainCount">,
  HelpText<"Check for leaks and improper reference count management">,
  DescFile<"RetainCountChecker.cpp">;

def ObjCGenericsChecker : Checker<"ObjCGenerics">,
  HelpText<"Check for type errors when using Objective-C generics">,
  DescFile<"DynamicTypePropagation.cpp">;

def ObjCDeallocChecker : Checker<"Dealloc">,
  HelpText<"Warn about Objective-C classes that lack a correct implementation of -dealloc">,
  DescFile<"CheckObjCDealloc.cpp">;

def ObjCSuperDeallocChecker : Checker<"SuperDealloc">,
  HelpText<"Warn about improper use of '[super dealloc]' in Objective-C">,
  DescFile<"ObjCSuperDeallocChecker.cpp">;

} // end "osx.cocoa"

let ParentPackage = Performance in {

def GCDAntipattern : Checker<"GCDAntipattern">,
  HelpText<"Check for performance anti-patterns when using Grand Central Dispatch">,
  DescFile<"GCDAntipatternChecker.cpp">;
} // end "optin.performance"

let ParentPackage = CocoaAlpha in {

def InstanceVariableInvalidation : Checker<"InstanceVariableInvalidation">,
  HelpText<"Check that the invalidatable instance variables are invalidated in the methods annotated with objc_instance_variable_invalidator">,
  DescFile<"IvarInvalidationChecker.cpp">;

def MissingInvalidationMethod : Checker<"MissingInvalidationMethod">,
  HelpText<"Check that the invalidation methods are present in classes that contain invalidatable instance variables">,
  DescFile<"IvarInvalidationChecker.cpp">;

def DirectIvarAssignment : Checker<"DirectIvarAssignment">,
  HelpText<"Check for direct assignments to instance variables">,
  DescFile<"DirectIvarAssignment.cpp">;

def DirectIvarAssignmentForAnnotatedFunctions : Checker<"DirectIvarAssignmentForAnnotatedFunctions">,
  HelpText<"Check for direct assignments to instance variables in the methods annotated with objc_no_direct_instance_variable_assignment">,
  DescFile<"DirectIvarAssignment.cpp">;

} // end "alpha.osx.cocoa"

let ParentPackage = CoreFoundation in {

def CFNumberChecker : Checker<"CFNumber">,
  HelpText<"Check for proper uses of CFNumber APIs">,
  DescFile<"BasicObjCFoundationChecks.cpp">;

def CFRetainReleaseChecker : Checker<"CFRetainRelease">,
  HelpText<"Check for null arguments to CFRetain/CFRelease/CFMakeCollectable">,
  DescFile<"BasicObjCFoundationChecks.cpp">;

def CFErrorChecker : Checker<"CFError">,
  HelpText<"Check usage of CFErrorRef* parameters">,
  DescFile<"NSErrorChecker.cpp">;
}

let ParentPackage = Containers in {
def ObjCContainersASTChecker : Checker<"PointerSizedValues">,
  HelpText<"Warns if 'CFArray', 'CFDictionary', 'CFSet' are created with non-pointer-size values">,
  DescFile<"ObjCContainersASTChecker.cpp">;

def ObjCContainersChecker : Checker<"OutOfBounds">,
  HelpText<"Checks for index out-of-bounds when using 'CFArray' API">,
  DescFile<"ObjCContainersChecker.cpp">;

}

let ParentPackage = LocalizabilityOptIn in {
def NonLocalizedStringChecker : Checker<"NonLocalizedStringChecker">,
  HelpText<"Warns about uses of non-localized NSStrings passed to UI methods expecting localized NSStrings">,
  DescFile<"LocalizationChecker.cpp">;

def EmptyLocalizationContextChecker : Checker<"EmptyLocalizationContextChecker">,
  HelpText<"Check that NSLocalizedString macros include a comment for context">,
  DescFile<"LocalizationChecker.cpp">;
}

let ParentPackage = LocalizabilityAlpha in {
def PluralMisuseChecker : Checker<"PluralMisuseChecker">,
  HelpText<"Warns against using one vs. many plural pattern in code when generating localized strings.">,
  DescFile<"LocalizationChecker.cpp">;
}

let ParentPackage = MPI in {
  def MPIChecker : Checker<"MPI-Checker">,
  HelpText<"Checks MPI code">,
  DescFile<"MPIChecker.cpp">;
}

//===----------------------------------------------------------------------===//
// Checkers for LLVM development.
//===----------------------------------------------------------------------===//

def LLVMConventionsChecker : Checker<"Conventions">,
  InPackage<LLVM>,
  HelpText<"Check code for LLVM codebase conventions">,
  DescFile<"LLVMConventionsChecker.cpp">;



//===----------------------------------------------------------------------===//
// Checkers modeling Google APIs.
//===----------------------------------------------------------------------===//

def GTestChecker : Checker<"GTest">,
  InPackage<GoogleAPIModeling>,
  HelpText<"Model gtest assertion APIs">,
  DescFile<"GTestChecker.cpp">;

//===----------------------------------------------------------------------===//
// Debugging checkers (for analyzer development).
//===----------------------------------------------------------------------===//

let ParentPackage = Debug in {

def AnalysisOrderChecker : Checker<"AnalysisOrder">,
  HelpText<"Print callbacks that are called during analysis in order">,
  DescFile<"AnalysisOrder.cpp">;

def DominatorsTreeDumper : Checker<"DumpDominators">,
  HelpText<"Print the dominance tree for a given CFG">,
  DescFile<"DebugCheckers.cpp">;

def LiveVariablesDumper : Checker<"DumpLiveVars">,
  HelpText<"Print results of live variable analysis">,
  DescFile<"DebugCheckers.cpp">;

def CFGViewer : Checker<"ViewCFG">,
  HelpText<"View Control-Flow Graphs using GraphViz">,
  DescFile<"DebugCheckers.cpp">;

def CFGDumper : Checker<"DumpCFG">,
  HelpText<"Display Control-Flow Graphs">,
  DescFile<"DebugCheckers.cpp">;

def CallGraphViewer : Checker<"ViewCallGraph">,
  HelpText<"View Call Graph using GraphViz">,
  DescFile<"DebugCheckers.cpp">;

def CallGraphDumper : Checker<"DumpCallGraph">,
  HelpText<"Display Call Graph">,
  DescFile<"DebugCheckers.cpp">;

def ConfigDumper : Checker<"ConfigDumper">,
  HelpText<"Dump config table">,
  DescFile<"DebugCheckers.cpp">;

def TraversalDumper : Checker<"DumpTraversal">,
  HelpText<"Print branch conditions as they are traversed by the engine">,
  DescFile<"TraversalChecker.cpp">;

def CallDumper : Checker<"DumpCalls">,
  HelpText<"Print calls as they are traversed by the engine">,
  DescFile<"TraversalChecker.cpp">;

def AnalyzerStatsChecker : Checker<"Stats">,
  HelpText<"Emit warnings with analyzer statistics">,
  DescFile<"AnalyzerStatsChecker.cpp">;

def TaintTesterChecker : Checker<"TaintTest">,
  HelpText<"Mark tainted symbols as such.">,
  DescFile<"TaintTesterChecker.cpp">;

def ExprInspectionChecker : Checker<"ExprInspection">,
  HelpText<"Check the analyzer's understanding of expressions">,
  DescFile<"ExprInspectionChecker.cpp">;

def ExplodedGraphViewer : Checker<"ViewExplodedGraph">,
  HelpText<"View Exploded Graphs using GraphViz">,
  DescFile<"DebugCheckers.cpp">;

} // end "debug"


//===----------------------------------------------------------------------===//
// Clone Detection
//===----------------------------------------------------------------------===//

let ParentPackage = CloneDetectionAlpha in {

def CloneChecker : Checker<"CloneChecker">,
  HelpText<"Reports similar pieces of code.">,
  DescFile<"CloneChecker.cpp">;

} // end "clone"

//===----------------------------------------------------------------------===//
// Portability checkers.
//===----------------------------------------------------------------------===//

let ParentPackage = PortabilityOptIn in {

def UnixAPIPortabilityChecker : Checker<"UnixAPI">,
  HelpText<"Finds implementation-defined behavior in UNIX/Posix functions">,
  DescFile<"UnixAPIChecker.cpp">;

} // end optin.portability
