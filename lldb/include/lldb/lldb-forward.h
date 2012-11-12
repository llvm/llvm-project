//===-- lldb-forward.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_lldb_forward_h_
#define LLDB_lldb_forward_h_

#if defined(__cplusplus)

#include <ciso646>  // detect C++ lib

#ifdef _LIBCPP_VERSION
#include <memory>
#define STD_SHARED_PTR(T) std::shared_ptr<T>
#define STD_WEAK_PTR(T) std::weak_ptr<T>
#define STD_ENABLE_SHARED_FROM_THIS(T) std::enable_shared_from_this<T>
#define STD_STATIC_POINTER_CAST(T,V) std::static_pointer_cast<T>(V)
#else
#include <tr1/memory>
#define STD_SHARED_PTR(T) std::tr1::shared_ptr<T>
#define STD_WEAK_PTR(T) std::tr1::weak_ptr<T>
#define STD_ENABLE_SHARED_FROM_THIS(T) std::tr1::enable_shared_from_this<T>
#define STD_STATIC_POINTER_CAST(T,V) std::tr1::static_pointer_cast<T>(V)
#endif

#include "lldb/Utility/SharingPtr.h"

//----------------------------------------------------------------------
// lldb forward declarations
//----------------------------------------------------------------------
namespace lldb_private {

class   ABI;
class   Address;
class   AddressImpl;
class   AddressRange;
class   AddressResolver;
class   ArchSpec;
class   Args;
class   ASTResultSynthesizer;
class   Baton;
class   Block;
class   Breakpoint;
class   BreakpointID;
class   BreakpointIDList;
class   BreakpointList;
class   BreakpointLocation;
class   BreakpointLocationCollection;
class   BreakpointLocationList;
class   BreakpointOptions;
class   BreakpointResolver;
class   BreakpointSite;
class   BreakpointSiteList;
class   BroadcastEventSpec;
class   Broadcaster;
class   BroadcasterManager;
class   CPPLanguageRuntime;
class   ClangASTContext;
class   ClangASTImporter;
class   ClangASTMetadata;
class   ClangASTSource;
class   ClangASTType;
class   ClangNamespaceDecl;
class   ClangExpression;
class   ClangExpressionDeclMap;
class   ClangExpressionParser;
class   ClangExpressionVariable;
class   ClangExpressionVariableList;
class   ClangExpressionVariableList;
class   ClangExpressionVariables;
class   ClangFunction;
class   ClangPersistentVariables;
class   ClangUserExpression;
class   ClangUtilityFunction;
class   CommandInterpreter;
class   CommandObject;
class   CommandReturnObject;
class   Communication;
class   CompileUnit;
class   Condition;
class   Connection;
class   ConnectionFileDescriptor;
class   ConstString;
class   CXXSyntheticChildren;
class   DWARFCallFrameInfo;
class   DWARFExpression;
class   DataBuffer;
class   DataEncoder;
class   DataExtractor;
class   Debugger;
class   Declaration;
class   Disassembler;
class   DynamicLibrary;
class   DynamicLoader;
class   EmulateInstruction;
class   Error;
class   EvaluateExpressionOptions;
class   Event;
class   EventData;
class   ExecutionContext;
class   ExecutionContextRef;
class   ExecutionContextRefLocker;
class   ExecutionContextScope;
class   FileSpec;
class   FileSpecList;
class   Flags;
class   TypeCategoryImpl;
class   FormatManager;
class   FuncUnwinders;
class   Function;
class   FunctionInfo;
class   InlineFunctionInfo;
class   InputReader;
class   Instruction;
class   InstructionList;
class   LanguageRuntime;
class   LineTable;
class   Listener;
class   Log;
class   LogChannel;
class   Mangled;
class   Module;
class   ModuleList;
class   ModuleSpec;
class   Mutex;
struct  NameSearchContext;
class   ObjCLanguageRuntime;
class   ObjectContainer;
class   OptionGroup;
class   OptionGroupPlatform;
class   ObjectFile;
class   OperatingSystem;
class   Options;
class   OptionValue;
class   OptionValueArch;
class   OptionValueArgs;
class   OptionValueArray;
class   OptionValueBoolean;
class   OptionValueDictionary;
class   OptionValueEnumeration;
class   OptionValueFileSpec;
class   OptionValueFileSpecList;
class   OptionValueFormat;
class   OptionValuePathMappings;
class   OptionValueProperties;
class   OptionValueRegex;
class   OptionValueSInt64;
class   OptionValueString;
class   OptionValueUInt64;
class   OptionValueUUID;
class   NamedOption;
class   PathMappingList;
class   Platform;
class   Process;
class   ProcessAttachInfo;
class   ProcessModID;
class   ProcessInfo;
class   ProcessInstanceInfo;
class   ProcessInstanceInfoList;
class   ProcessInstanceInfoMatch;
class   ProcessLaunchInfo;
class   Property;
struct  PropertyDefinition;
class   PythonDataArray;
class   PythonDataDictionary;
class   PythonDataInteger;
class   PythonDataObject;
class   PythonDataString;
class   RegisterContext;
class   RegisterLocation;
class   RegisterLocationList;
class   RegisterValue;
class   RegularExpression;
class   Scalar;
class   ScriptInterpreter;
class   ScriptInterpreterObject;
#ifndef LLDB_DISABLE_PYTHON
class   ScriptInterpreterPython;
struct  ScriptSummaryFormat;
#endif
class   SearchFilter;
class   Section;
class   SectionImpl;
class   SectionList;
class   Settings;
class   SourceManager;
class   SourceManagerImpl;
class   StackFrame;
class   StackFrameImpl;
class   StackFrameList;
class   StackID;
class   StopInfo;
class   Stoppoint;
class   StoppointCallbackContext;
class   StoppointLocation;
class   Stream;
template <unsigned N> class StreamBuffer;
class   StreamFile;
class   StreamString;
class   StringList;
struct  StringSummaryFormat;
class   TypeSummaryImpl;
class   Symbol;
class   SymbolContext;
class   SymbolContextList;
class   SymbolContextScope;
class   SymbolContextSpecifier;
class   SymbolFile;
class   SymbolFileType;
class   SymbolVendor;
class   Symtab;
class   SyntheticChildren;
class   SyntheticChildrenFrontEnd;
class   TypeFilterImpl;
#ifndef LLDB_DISABLE_PYTHON
class   TypeSyntheticImpl;
#endif
class   Target;
class   TargetList;
class   Thread;
class   ThreadList;
class   ThreadPlan;
class   ThreadPlanBase;
class   ThreadPlanRunToAddress;
class   ThreadPlanStepInstruction;
class   ThreadPlanStepOut;
class   ThreadPlanStepOverBreakpoint;
class   ThreadPlanStepRange;
class   ThreadPlanStepThrough;
class   ThreadPlanTracer;
class   ThreadSpec;
class   TimeValue;
class   Type;
class   TypeImpl;
class   TypeAndOrName;
class   TypeList;
class   TypeListImpl;
class   TypeMemberImpl;
class   TypeNameSpecifierImpl;
class   UUID;
class   Unwind;
class   UnwindAssembly;
class   UnwindPlan;
class   UnwindTable;
class   VMRange;
class   Value;
class   TypeFormatImpl;
class   ValueList;
class   ValueObject;
class   ValueObjectChild;
class   ValueObjectConstResult;
class   ValueObjectConstResultChild;
class   ValueObjectConstResultImpl;
class   ValueObjectList;
class   Variable;
class   VariableList;
class   Watchpoint;
class   WatchpointList;
class   WatchpointOptions;
struct  LineEntry;

} // namespace lldb_private

//----------------------------------------------------------------------
// lldb forward declarations
//----------------------------------------------------------------------
namespace lldb {
    
    typedef STD_SHARED_PTR(lldb_private::ABI) ABISP;
    typedef STD_SHARED_PTR(lldb_private::Baton) BatonSP;
    typedef STD_SHARED_PTR(lldb_private::Block) BlockSP;
    typedef STD_SHARED_PTR(lldb_private::Breakpoint) BreakpointSP;
    typedef STD_WEAK_PTR(  lldb_private::Breakpoint) BreakpointWP;
    typedef STD_SHARED_PTR(lldb_private::BreakpointSite) BreakpointSiteSP;
    typedef STD_WEAK_PTR(  lldb_private::BreakpointSite) BreakpointSiteWP;
    typedef STD_SHARED_PTR(lldb_private::BreakpointLocation) BreakpointLocationSP;
    typedef STD_WEAK_PTR(  lldb_private::BreakpointLocation) BreakpointLocationWP;
    typedef STD_SHARED_PTR(lldb_private::BreakpointResolver) BreakpointResolverSP;
    typedef STD_SHARED_PTR(lldb_private::Broadcaster) BroadcasterSP;
    typedef STD_SHARED_PTR(lldb_private::ClangExpressionVariable) ClangExpressionVariableSP;
    typedef STD_SHARED_PTR(lldb_private::CommandObject) CommandObjectSP;
    typedef STD_SHARED_PTR(lldb_private::Communication) CommunicationSP;
    typedef STD_SHARED_PTR(lldb_private::Connection) ConnectionSP;
    typedef STD_SHARED_PTR(lldb_private::CompileUnit) CompUnitSP;
    typedef STD_SHARED_PTR(lldb_private::DataBuffer) DataBufferSP;
    typedef STD_SHARED_PTR(lldb_private::DataExtractor) DataExtractorSP;
    typedef STD_SHARED_PTR(lldb_private::Debugger) DebuggerSP;
    typedef STD_WEAK_PTR(  lldb_private::Debugger) DebuggerWP;
    typedef STD_SHARED_PTR(lldb_private::Disassembler) DisassemblerSP;
    typedef STD_SHARED_PTR(lldb_private::DynamicLibrary) DynamicLibrarySP;
    typedef STD_SHARED_PTR(lldb_private::DynamicLoader) DynamicLoaderSP;
    typedef STD_SHARED_PTR(lldb_private::Event) EventSP;
    typedef STD_SHARED_PTR(lldb_private::ExecutionContextRef) ExecutionContextRefSP;
    typedef STD_SHARED_PTR(lldb_private::Function) FunctionSP;
    typedef STD_SHARED_PTR(lldb_private::FuncUnwinders) FuncUnwindersSP;
    typedef STD_SHARED_PTR(lldb_private::InlineFunctionInfo) InlineFunctionInfoSP;
    typedef STD_SHARED_PTR(lldb_private::InputReader) InputReaderSP;
    typedef STD_SHARED_PTR(lldb_private::Instruction) InstructionSP;
    typedef STD_SHARED_PTR(lldb_private::LanguageRuntime) LanguageRuntimeSP;
    typedef STD_SHARED_PTR(lldb_private::LineTable) LineTableSP;
    typedef STD_SHARED_PTR(lldb_private::Listener) ListenerSP;
    typedef STD_SHARED_PTR(lldb_private::Log) LogSP;
    typedef STD_SHARED_PTR(lldb_private::LogChannel) LogChannelSP;
    typedef STD_SHARED_PTR(lldb_private::Module) ModuleSP;
    typedef STD_WEAK_PTR(  lldb_private::Module) ModuleWP;
    typedef STD_SHARED_PTR(lldb_private::ObjectFile) ObjectFileSP;
    typedef STD_WEAK_PTR(  lldb_private::ObjectFile) ObjectFileWP;
    typedef STD_SHARED_PTR(lldb_private::OptionValue) OptionValueSP;
    typedef STD_WEAK_PTR(  lldb_private::OptionValue) OptionValueWP;
    typedef STD_SHARED_PTR(lldb_private::OptionValueArch) OptionValueArchSP;
    typedef STD_SHARED_PTR(lldb_private::OptionValueArgs) OptionValueArgsSP;
    typedef STD_SHARED_PTR(lldb_private::OptionValueArray) OptionValueArraySP;
    typedef STD_SHARED_PTR(lldb_private::OptionValueBoolean) OptionValueBooleanSP;
    typedef STD_SHARED_PTR(lldb_private::OptionValueDictionary) OptionValueDictionarySP;
    typedef STD_SHARED_PTR(lldb_private::OptionValueFileSpec) OptionValueFileSpecSP;
    typedef STD_SHARED_PTR(lldb_private::OptionValueFileSpecList) OptionValueFileSpecListSP;
    typedef STD_SHARED_PTR(lldb_private::OptionValueFormat) OptionValueFormatSP;
    typedef STD_SHARED_PTR(lldb_private::OptionValuePathMappings) OptionValuePathMappingsSP;
    typedef STD_SHARED_PTR(lldb_private::OptionValueProperties) OptionValuePropertiesSP;
//  typedef STD_WEAK_PTR(  lldb_private::OptionValueProperties) OptionValuePropertiesWP;
    typedef STD_SHARED_PTR(lldb_private::OptionValueRegex) OptionValueRegexSP;
    typedef STD_SHARED_PTR(lldb_private::OptionValueSInt64) OptionValueSInt64SP;
    typedef STD_SHARED_PTR(lldb_private::OptionValueString) OptionValueStringSP;
    typedef STD_SHARED_PTR(lldb_private::OptionValueUInt64) OptionValueUInt64SP;
    typedef STD_SHARED_PTR(lldb_private::OptionValueUUID) OptionValueUUIDSP;
    typedef STD_SHARED_PTR(lldb_private::Platform) PlatformSP;
    typedef STD_SHARED_PTR(lldb_private::Process) ProcessSP;
    typedef STD_SHARED_PTR(lldb_private::ProcessAttachInfo) ProcessAttachInfoSP;
    typedef STD_SHARED_PTR(lldb_private::ProcessLaunchInfo) ProcessLaunchInfoSP;
    typedef STD_WEAK_PTR(  lldb_private::Process) ProcessWP;
    typedef STD_SHARED_PTR(lldb_private::Property) PropertySP;
    typedef STD_SHARED_PTR(lldb_private::RegisterContext) RegisterContextSP;
    typedef STD_SHARED_PTR(lldb_private::RegularExpression) RegularExpressionSP;
    typedef STD_SHARED_PTR(lldb_private::ScriptInterpreterObject) ScriptInterpreterObjectSP;
#ifndef LLDB_DISABLE_PYTHON
    typedef STD_SHARED_PTR(lldb_private::ScriptSummaryFormat) ScriptSummaryFormatSP;
#endif // #ifndef LLDB_DISABLE_PYTHON
    typedef STD_SHARED_PTR(lldb_private::Section) SectionSP;
    typedef STD_WEAK_PTR(  lldb_private::Section) SectionWP;
    typedef STD_SHARED_PTR(lldb_private::SearchFilter) SearchFilterSP;
    typedef STD_SHARED_PTR(lldb_private::Settings) SettingsSP;
    typedef STD_SHARED_PTR(lldb_private::StackFrame) StackFrameSP;
    typedef STD_WEAK_PTR(  lldb_private::StackFrame) StackFrameWP;
    typedef STD_SHARED_PTR(lldb_private::StackFrameList) StackFrameListSP;
    typedef STD_SHARED_PTR(lldb_private::StopInfo) StopInfoSP;
    typedef STD_SHARED_PTR(lldb_private::StoppointLocation) StoppointLocationSP;
    typedef STD_SHARED_PTR(lldb_private::Stream) StreamSP;
    typedef STD_SHARED_PTR(lldb_private::StringSummaryFormat) StringTypeSummaryImplSP;
    typedef STD_SHARED_PTR(lldb_private::SymbolFile) SymbolFileSP;
    typedef STD_SHARED_PTR(lldb_private::SymbolFileType) SymbolFileTypeSP;
    typedef STD_WEAK_PTR(  lldb_private::SymbolFileType) SymbolFileTypeWP;
    typedef STD_SHARED_PTR(lldb_private::SymbolContextSpecifier) SymbolContextSpecifierSP;
    typedef STD_SHARED_PTR(lldb_private::SyntheticChildren) SyntheticChildrenSP;
    typedef STD_SHARED_PTR(lldb_private::SyntheticChildrenFrontEnd) SyntheticChildrenFrontEndSP;
    typedef STD_SHARED_PTR(lldb_private::Target) TargetSP;
    typedef STD_WEAK_PTR(  lldb_private::Target) TargetWP;
    typedef STD_SHARED_PTR(lldb_private::Thread) ThreadSP;
    typedef STD_WEAK_PTR(  lldb_private::Thread) ThreadWP;
    typedef STD_SHARED_PTR(lldb_private::ThreadPlan) ThreadPlanSP;
    typedef STD_SHARED_PTR(lldb_private::ThreadPlanTracer) ThreadPlanTracerSP;
    typedef STD_SHARED_PTR(lldb_private::Type) TypeSP;
    typedef STD_WEAK_PTR(  lldb_private::Type) TypeWP;
    typedef STD_SHARED_PTR(lldb_private::TypeCategoryImpl) TypeCategoryImplSP;
    typedef STD_SHARED_PTR(lldb_private::TypeImpl) TypeImplSP;
    typedef STD_SHARED_PTR(lldb_private::TypeFilterImpl) TypeFilterImplSP;
    typedef STD_SHARED_PTR(lldb_private::TypeFormatImpl) TypeFormatImplSP;
    typedef STD_SHARED_PTR(lldb_private::TypeNameSpecifierImpl) TypeNameSpecifierImplSP;
    typedef STD_SHARED_PTR(lldb_private::TypeSummaryImpl) TypeSummaryImplSP;
#ifndef LLDB_DISABLE_PYTHON
    typedef STD_SHARED_PTR(lldb_private::TypeSyntheticImpl) TypeSyntheticImplSP;
#endif
    typedef STD_SHARED_PTR(lldb_private::UnwindPlan) UnwindPlanSP;
    typedef lldb_private::SharingPtr<lldb_private::ValueObject> ValueObjectSP;
    typedef STD_SHARED_PTR(lldb_private::Value) ValueSP;
    typedef STD_SHARED_PTR(lldb_private::ValueList) ValueListSP;
    typedef STD_SHARED_PTR(lldb_private::Variable) VariableSP;
    typedef STD_SHARED_PTR(lldb_private::VariableList) VariableListSP;
    typedef STD_SHARED_PTR(lldb_private::ValueObjectList) ValueObjectListSP;
    typedef STD_SHARED_PTR(lldb_private::Watchpoint) WatchpointSP;
    
} // namespace lldb


#endif  // #if defined(__cplusplus)
#endif  // LLDB_lldb_forward_h_
