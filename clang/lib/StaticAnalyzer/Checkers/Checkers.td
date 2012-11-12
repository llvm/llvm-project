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

def Alpha : Package<"alpha">;

def Core : Package<"core">;
def CoreBuiltin : Package<"builtin">, InPackage<Core>;
def CoreUninitialized  : Package<"uninitialized">, InPackage<Core>;
def CoreAlpha : Package<"core">, InPackage<Alpha>, Hidden;

def Cplusplus : Package<"cplusplus">;
def CplusplusAlpha : Package<"cplusplus">, InPackage<Alpha>, Hidden;

def DeadCode : Package<"deadcode">;
def DeadCodeAlpha : Package<"deadcode">, InPackage<Alpha>, Hidden;

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
def Cocoa : Package<"cocoa">, InPackage<OSX>;
def CocoaAlpha : Package<"cocoa">, InPackage<OSXAlpha>, Hidden;
def CoreFoundation : Package<"coreFoundation">, InPackage<OSX>;
def Containers : Package<"containers">, InPackage<CoreFoundation>;

def LLVM : Package<"llvm">;
def Debug : Package<"debug">;

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

def AttrNonNullChecker : Checker<"AttributeNonNull">,
  HelpText<"Check for null pointers passed as arguments to a function whose arguments are marked with the 'nonnull' attribute">,
  DescFile<"AttrNonNullChecker.cpp">;

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

} // end "alpha.core"

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

let ParentPackage = CplusplusAlpha in {

def VirtualCallChecker : Checker<"VirtualCall">,
  HelpText<"Check virtual function calls during construction or destruction">, 
  DescFile<"VirtualCallChecker.cpp">;

} // end: "alpha.cplusplus"

//===----------------------------------------------------------------------===//
// Deadcode checkers.
//===----------------------------------------------------------------------===//

let ParentPackage = DeadCode in {

def DeadStoresChecker : Checker<"DeadStores">,
  HelpText<"Check for values stored to variables that are never read afterwards">,
  DescFile<"DeadStoresChecker.cpp">;
} // end DeadCode

let ParentPackage = DeadCodeAlpha in {

def IdempotentOperationChecker : Checker<"IdempotentOperations">,
  HelpText<"Warn about idempotent operations">,
  DescFile<"IdempotentOperationChecker.cpp">;

def UnreachableCodeChecker : Checker<"UnreachableCode">,
  HelpText<"Check unreachable code">,
  DescFile<"UnreachableCodeChecker.cpp">;

} // end "alpha.deadcode"

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

def UnixAPIChecker : Checker<"API">,
  HelpText<"Check calls to various UNIX/Posix functions">,
  DescFile<"UnixAPIChecker.cpp">;

def MallocPessimistic : Checker<"Malloc">,
  HelpText<"Check for memory leaks, double free, and use-after-free problems.">,
  DescFile<"MallocChecker.cpp">;
  
def MallocSizeofChecker : Checker<"MallocSizeof">,
  HelpText<"Check for dubious malloc arguments involving sizeof">,
  DescFile<"MallocSizeofChecker.cpp">;
  
} // end "unix"

let ParentPackage = UnixAlpha in {

def ChrootChecker : Checker<"Chroot">,
  HelpText<"Check improper use of chroot">,
  DescFile<"ChrootChecker.cpp">;

def MallocOptimistic : Checker<"MallocWithAnnotations">,
  HelpText<"Check for memory leaks, double free, and use-after-free problems. Assumes that all user-defined functions which might free a pointer are annotated.">,
  DescFile<"MallocChecker.cpp">;

def PthreadLockChecker : Checker<"PthreadLock">,
  HelpText<"Simple lock -> unlock checker">,
  DescFile<"PthreadLockChecker.cpp">;

def StreamChecker : Checker<"Stream">,
  HelpText<"Check stream handling functions">,
  DescFile<"StreamChecker.cpp">;

def SimpleStreamChecker : Checker<"SimpleStream">,
  HelpText<"Check for misuses of stream APIs">,
  DescFile<"SimpleStreamChecker.cpp">;

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

def MacOSXAPIChecker : Checker<"API">,
  InPackage<OSX>,
  HelpText<"Check for proper uses of various Mac OS X APIs">,
  DescFile<"MacOSXAPIChecker.cpp">;

def MacOSKeychainAPIChecker : Checker<"SecKeychainAPI">,
  InPackage<OSX>,
  HelpText<"Check for proper uses of Secure Keychain APIs">,
  DescFile<"MacOSKeychainAPIChecker.cpp">;

} // end "macosx"

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

def NSErrorChecker : Checker<"NSError">,
  HelpText<"Check usage of NSError** parameters">,
  DescFile<"NSErrorChecker.cpp">;

def RetainCountChecker : Checker<"RetainCount">,
  HelpText<"Check for leaks and improper reference count management">,
  DescFile<"RetainCountChecker.cpp">;

} // end "osx.cocoa"

let ParentPackage = CocoaAlpha in {

def ObjCDeallocChecker : Checker<"Dealloc">,
  HelpText<"Warn about Objective-C classes that lack a correct implementation of -dealloc">,
  DescFile<"CheckObjCDealloc.cpp">;

def IvarInvalidationChecker : Checker<"InstanceVariableInvalidation">,
  HelpText<"Check that the invalidatable instance variables are invalidated in the methods annotated with objc_instance_variable_invalidator">,
  DescFile<"IvarInvalidationChecker.cpp">;

def DirectIvarAssignment : Checker<"DirectIvarAssignment">,
  HelpText<"Check that the invalidatable instance variables are invalidated in the methods annotated with objc_instance_variable_invalidator">,
  DescFile<"DirectIvarAssignment.cpp">;

def ObjCSuperCallChecker : Checker<"MissingSuperCall">,
  HelpText<"Warn about Objective-C methods that lack a necessary call to super">,
  DescFile<"ObjCMissingSuperCallChecker.cpp">;

} // end "alpha.osx.cocoa"

let ParentPackage = CoreFoundation in {

def CFNumberCreateChecker : Checker<"CFNumber">,
  HelpText<"Check for proper uses of CFNumberCreate">,
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
//===----------------------------------------------------------------------===//
// Checkers for LLVM development.
//===----------------------------------------------------------------------===//

def LLVMConventionsChecker : Checker<"Conventions">,
  InPackage<LLVM>,
  HelpText<"Check code for LLVM codebase conventions">,
  DescFile<"LLVMConventionsChecker.cpp">;

//===----------------------------------------------------------------------===//
// Debugging checkers (for analyzer development).
//===----------------------------------------------------------------------===//

let ParentPackage = Debug in {

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

} // end "debug"

