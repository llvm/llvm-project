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
class   IRExecutionUnit;
class   LanguageRuntime;
class   LineTable;
class   Listener;
class   Log;
class   LogChannel;
class   Mangled;
class   Materializer;
class   Module;
class   ModuleList;
class   ModuleSpec;
class   ModuleSpecList;
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
class   PythonArray;
class   PythonDictionary;
class   PythonInteger;
class   PythonObject;
class   PythonString;
class   RegisterContext;
class   RegisterLocation;
class   RegisterLocationList;
class   RegisterValue;
class   RegularExpression;
class   Scalar;
class   ScriptInterpreter;
class   ScriptInterpreterLocker;
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
class   ScriptedSyntheticChildren;
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
class   TypeCategoryMap;
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
    
    typedef std::shared_ptr<lldb_private::ABI> ABISP;
    typedef std::shared_ptr<lldb_private::Baton> BatonSP;
    typedef std::shared_ptr<lldb_private::Block> BlockSP;
    typedef std::shared_ptr<lldb_private::Breakpoint> BreakpointSP;
    typedef std::weak_ptr<lldb_private::Breakpoint> BreakpointWP;
    typedef std::shared_ptr<lldb_private::BreakpointSite> BreakpointSiteSP;
    typedef std::weak_ptr<lldb_private::BreakpointSite> BreakpointSiteWP;
    typedef std::shared_ptr<lldb_private::BreakpointLocation> BreakpointLocationSP;
    typedef std::weak_ptr<lldb_private::BreakpointLocation> BreakpointLocationWP;
    typedef std::shared_ptr<lldb_private::BreakpointResolver> BreakpointResolverSP;
    typedef std::shared_ptr<lldb_private::Broadcaster> BroadcasterSP;
    typedef std::shared_ptr<lldb_private::ClangExpressionVariable> ClangExpressionVariableSP;
    typedef std::shared_ptr<lldb_private::CommandObject> CommandObjectSP;
    typedef std::shared_ptr<lldb_private::Communication> CommunicationSP;
    typedef std::shared_ptr<lldb_private::Connection> ConnectionSP;
    typedef std::shared_ptr<lldb_private::CompileUnit> CompUnitSP;
    typedef std::shared_ptr<lldb_private::DataBuffer> DataBufferSP;
    typedef std::shared_ptr<lldb_private::DataExtractor> DataExtractorSP;
    typedef std::shared_ptr<lldb_private::Debugger> DebuggerSP;
    typedef std::weak_ptr<lldb_private::Debugger> DebuggerWP;
    typedef std::shared_ptr<lldb_private::Disassembler> DisassemblerSP;
    typedef std::shared_ptr<lldb_private::DynamicLibrary> DynamicLibrarySP;
    typedef std::shared_ptr<lldb_private::DynamicLoader> DynamicLoaderSP;
    typedef std::shared_ptr<lldb_private::Event> EventSP;
    typedef std::shared_ptr<lldb_private::ExecutionContextRef> ExecutionContextRefSP;
    typedef std::shared_ptr<lldb_private::Function> FunctionSP;
    typedef std::shared_ptr<lldb_private::FuncUnwinders> FuncUnwindersSP;
    typedef std::shared_ptr<lldb_private::InlineFunctionInfo> InlineFunctionInfoSP;
    typedef std::shared_ptr<lldb_private::InputReader> InputReaderSP;
    typedef std::shared_ptr<lldb_private::Instruction> InstructionSP;
    typedef std::shared_ptr<lldb_private::LanguageRuntime> LanguageRuntimeSP;
    typedef std::shared_ptr<lldb_private::LineTable> LineTableSP;
    typedef std::shared_ptr<lldb_private::Listener> ListenerSP;
    typedef std::shared_ptr<lldb_private::LogChannel> LogChannelSP;
    typedef std::shared_ptr<lldb_private::Module> ModuleSP;
    typedef std::weak_ptr<lldb_private::Module> ModuleWP;
    typedef std::shared_ptr<lldb_private::ObjectFile> ObjectFileSP;
    typedef std::weak_ptr<lldb_private::ObjectFile> ObjectFileWP;
    typedef std::shared_ptr<lldb_private::OptionValue> OptionValueSP;
    typedef std::weak_ptr<lldb_private::OptionValue> OptionValueWP;
    typedef std::shared_ptr<lldb_private::OptionValueArch> OptionValueArchSP;
    typedef std::shared_ptr<lldb_private::OptionValueArgs> OptionValueArgsSP;
    typedef std::shared_ptr<lldb_private::OptionValueArray> OptionValueArraySP;
    typedef std::shared_ptr<lldb_private::OptionValueBoolean> OptionValueBooleanSP;
    typedef std::shared_ptr<lldb_private::OptionValueDictionary> OptionValueDictionarySP;
    typedef std::shared_ptr<lldb_private::OptionValueFileSpec> OptionValueFileSpecSP;
    typedef std::shared_ptr<lldb_private::OptionValueFileSpecList> OptionValueFileSpecListSP;
    typedef std::shared_ptr<lldb_private::OptionValueFormat> OptionValueFormatSP;
    typedef std::shared_ptr<lldb_private::OptionValuePathMappings> OptionValuePathMappingsSP;
    typedef std::shared_ptr<lldb_private::OptionValueProperties> OptionValuePropertiesSP;
    typedef std::shared_ptr<lldb_private::OptionValueRegex> OptionValueRegexSP;
    typedef std::shared_ptr<lldb_private::OptionValueSInt64> OptionValueSInt64SP;
    typedef std::shared_ptr<lldb_private::OptionValueString> OptionValueStringSP;
    typedef std::shared_ptr<lldb_private::OptionValueUInt64> OptionValueUInt64SP;
    typedef std::shared_ptr<lldb_private::OptionValueUUID> OptionValueUUIDSP;
    typedef std::shared_ptr<lldb_private::Platform> PlatformSP;
    typedef std::shared_ptr<lldb_private::Process> ProcessSP;
    typedef std::shared_ptr<lldb_private::ProcessAttachInfo> ProcessAttachInfoSP;
    typedef std::shared_ptr<lldb_private::ProcessLaunchInfo> ProcessLaunchInfoSP;
    typedef std::weak_ptr<lldb_private::Process> ProcessWP;
    typedef std::shared_ptr<lldb_private::Property> PropertySP;
    typedef std::shared_ptr<lldb_private::RegisterContext> RegisterContextSP;
    typedef std::shared_ptr<lldb_private::RegularExpression> RegularExpressionSP;
    typedef std::shared_ptr<lldb_private::ScriptInterpreterObject> ScriptInterpreterObjectSP;
#ifndef LLDB_DISABLE_PYTHON
    typedef std::shared_ptr<lldb_private::ScriptSummaryFormat> ScriptSummaryFormatSP;
#endif // #ifndef LLDB_DISABLE_PYTHON
    typedef std::shared_ptr<lldb_private::Section> SectionSP;
    typedef std::weak_ptr<lldb_private::Section> SectionWP;
    typedef std::shared_ptr<lldb_private::SearchFilter> SearchFilterSP;
    typedef std::shared_ptr<lldb_private::Settings> SettingsSP;
    typedef std::shared_ptr<lldb_private::StackFrame> StackFrameSP;
    typedef std::weak_ptr<lldb_private::StackFrame> StackFrameWP;
    typedef std::shared_ptr<lldb_private::StackFrameList> StackFrameListSP;
    typedef std::shared_ptr<lldb_private::StopInfo> StopInfoSP;
    typedef std::shared_ptr<lldb_private::StoppointLocation> StoppointLocationSP;
    typedef std::shared_ptr<lldb_private::Stream> StreamSP;
    typedef std::weak_ptr<lldb_private::Stream> StreamWP;
    typedef std::shared_ptr<lldb_private::StringSummaryFormat> StringTypeSummaryImplSP;
    typedef std::shared_ptr<lldb_private::SymbolFile> SymbolFileSP;
    typedef std::shared_ptr<lldb_private::SymbolFileType> SymbolFileTypeSP;
    typedef std::weak_ptr<lldb_private::SymbolFileType> SymbolFileTypeWP;
    typedef std::shared_ptr<lldb_private::SymbolContextSpecifier> SymbolContextSpecifierSP;
    typedef std::shared_ptr<lldb_private::SyntheticChildren> SyntheticChildrenSP;
    typedef std::shared_ptr<lldb_private::SyntheticChildrenFrontEnd> SyntheticChildrenFrontEndSP;
    typedef std::shared_ptr<lldb_private::Target> TargetSP;
    typedef std::weak_ptr<lldb_private::Target> TargetWP;
    typedef std::shared_ptr<lldb_private::Thread> ThreadSP;
    typedef std::weak_ptr<lldb_private::Thread> ThreadWP;
    typedef std::shared_ptr<lldb_private::ThreadPlan> ThreadPlanSP;
    typedef std::shared_ptr<lldb_private::ThreadPlanTracer> ThreadPlanTracerSP;
    typedef std::shared_ptr<lldb_private::Type> TypeSP;
    typedef std::weak_ptr<lldb_private::Type> TypeWP;
    typedef std::shared_ptr<lldb_private::TypeCategoryImpl> TypeCategoryImplSP;
    typedef std::shared_ptr<lldb_private::TypeImpl> TypeImplSP;
    typedef std::shared_ptr<lldb_private::TypeFilterImpl> TypeFilterImplSP;
    typedef std::shared_ptr<lldb_private::TypeFormatImpl> TypeFormatImplSP;
    typedef std::shared_ptr<lldb_private::TypeNameSpecifierImpl> TypeNameSpecifierImplSP;
    typedef std::shared_ptr<lldb_private::TypeSummaryImpl> TypeSummaryImplSP;
#ifndef LLDB_DISABLE_PYTHON
    typedef std::shared_ptr<lldb_private::ScriptedSyntheticChildren> ScriptedSyntheticChildrenSP;
#endif
    typedef std::shared_ptr<lldb_private::UnwindPlan> UnwindPlanSP;
    typedef lldb_private::SharingPtr<lldb_private::ValueObject> ValueObjectSP;
    typedef std::shared_ptr<lldb_private::Value> ValueSP;
    typedef std::shared_ptr<lldb_private::ValueList> ValueListSP;
    typedef std::shared_ptr<lldb_private::Variable> VariableSP;
    typedef std::shared_ptr<lldb_private::VariableList> VariableListSP;
    typedef std::shared_ptr<lldb_private::ValueObjectList> ValueObjectListSP;
    typedef std::shared_ptr<lldb_private::Watchpoint> WatchpointSP;
    
} // namespace lldb


#endif  // #if defined(__cplusplus)
#endif  // LLDB_lldb_forward_h_
