//===-- SystemInitializerFull.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if !defined(LLDB_DISABLE_PYTHON)
#include "Plugins/ScriptInterpreter/Python/lldb-python.h"
#endif

#include "SystemInitializerFull.h"

#include "lldb/API/SBCommandInterpreter.h"

#if !defined(LLDB_DISABLE_PYTHON)
#include "Plugins/ScriptInterpreter/Python/ScriptInterpreterPython.h"
#endif

#include "lldb/Core/Debugger.h"
#include "lldb/Host/Host.h"
#include "lldb/Initialization/SystemInitializerCommon.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/GoASTContext.h"
#include "lldb/Symbol/JavaASTContext.h"
#include "lldb/Symbol/OCamlASTContext.h"
#include "lldb/Utility/Timer.h"

#include "Plugins/ABI/MacOSX-arm/ABIMacOSX_arm.h"
#include "Plugins/ABI/MacOSX-arm64/ABIMacOSX_arm64.h"
#include "Plugins/ABI/MacOSX-i386/ABIMacOSX_i386.h"
#include "Plugins/ABI/SysV-arm/ABISysV_arm.h"
#include "Plugins/ABI/SysV-arm64/ABISysV_arm64.h"
#include "Plugins/ABI/SysV-hexagon/ABISysV_hexagon.h"
#include "Plugins/ABI/SysV-i386/ABISysV_i386.h"
#include "Plugins/ABI/SysV-mips/ABISysV_mips.h"
#include "Plugins/ABI/SysV-mips64/ABISysV_mips64.h"
#include "Plugins/ABI/SysV-ppc/ABISysV_ppc.h"
#include "Plugins/ABI/SysV-ppc64/ABISysV_ppc64.h"
#include "Plugins/ABI/SysV-s390x/ABISysV_s390x.h"
#include "Plugins/ABI/SysV-x86_64/ABISysV_x86_64.h"
#include "Plugins/Architecture/Arm/ArchitectureArm.h"
#include "Plugins/Architecture/PPC64/ArchitecturePPC64.h"
#include "Plugins/Disassembler/llvm/DisassemblerLLVMC.h"
#include "Plugins/DynamicLoader/MacOSX-DYLD/DynamicLoaderMacOS.h"
#include "Plugins/DynamicLoader/MacOSX-DYLD/DynamicLoaderMacOSXDYLD.h"
#include "Plugins/DynamicLoader/POSIX-DYLD/DynamicLoaderPOSIXDYLD.h"
#include "Plugins/DynamicLoader/Static/DynamicLoaderStatic.h"
#include "Plugins/DynamicLoader/Windows-DYLD/DynamicLoaderWindowsDYLD.h"
#include "Plugins/Instruction/ARM64/EmulateInstructionARM64.h"
#include "Plugins/Instruction/PPC64/EmulateInstructionPPC64.h"
#include "Plugins/InstrumentationRuntime/ASan/ASanRuntime.h"
#include "Plugins/InstrumentationRuntime/MainThreadChecker/MainThreadCheckerRuntime.h"
#include "Plugins/InstrumentationRuntime/TSan/TSanRuntime.h"
#include "Plugins/InstrumentationRuntime/UBSan/UBSanRuntime.h"
#include "Plugins/JITLoader/GDB/JITLoaderGDB.h"
#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"
#include "Plugins/Language/Go/GoLanguage.h"
#include "Plugins/Language/Java/JavaLanguage.h"
#include "Plugins/Language/OCaml/OCamlLanguage.h"
#include "Plugins/Language/ObjC/ObjCLanguage.h"
#include "Plugins/Language/ObjCPlusPlus/ObjCPlusPlusLanguage.h"
#include "Plugins/LanguageRuntime/CPlusPlus/ItaniumABI/ItaniumABILanguageRuntime.h"
#include "Plugins/LanguageRuntime/Go/GoLanguageRuntime.h"
#include "Plugins/LanguageRuntime/Java/JavaLanguageRuntime.h"
#include "Plugins/LanguageRuntime/ObjC/AppleObjCRuntime/AppleObjCRuntimeV1.h"
#include "Plugins/LanguageRuntime/ObjC/AppleObjCRuntime/AppleObjCRuntimeV2.h"
#include "Plugins/LanguageRuntime/RenderScript/RenderScriptRuntime/RenderScriptRuntime.h"
#include "Plugins/MemoryHistory/asan/MemoryHistoryASan.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "Plugins/OperatingSystem/Go/OperatingSystemGo.h"
#include "Plugins/OperatingSystem/Python/OperatingSystemPython.h"
#include "Plugins/Platform/Android/PlatformAndroid.h"
#include "Plugins/Platform/FreeBSD/PlatformFreeBSD.h"
#include "Plugins/Platform/Kalimba/PlatformKalimba.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteiOS.h"
#include "Plugins/Platform/NetBSD/PlatformNetBSD.h"
#include "Plugins/Platform/OpenBSD/PlatformOpenBSD.h"
#include "Plugins/Platform/Windows/PlatformWindows.h"
#include "Plugins/Platform/gdb-server/PlatformRemoteGDBServer.h"
#include "Plugins/Process/elf-core/ProcessElfCore.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"
#include "Plugins/Process/mach-core/ProcessMachCore.h"
#include "Plugins/Process/minidump/ProcessMinidump.h"
#include "Plugins/ScriptInterpreter/None/ScriptInterpreterNone.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARFDebugMap.h"
#include "Plugins/SymbolFile/PDB/SymbolFilePDB.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "Plugins/SymbolVendor/ELF/SymbolVendorELF.h"
#include "Plugins/SystemRuntime/MacOSX/SystemRuntimeMacOSX.h"
#include "Plugins/UnwindAssembly/InstEmulation/UnwindAssemblyInstEmulation.h"
#include "Plugins/UnwindAssembly/x86/UnwindAssembly-x86.h"

#if defined(__APPLE__)
#include "Plugins/DynamicLoader/Darwin-Kernel/DynamicLoaderDarwinKernel.h"
#include "Plugins/Platform/MacOSX/PlatformAppleTVSimulator.h"
#include "Plugins/Platform/MacOSX/PlatformAppleWatchSimulator.h"
#include "Plugins/Platform/MacOSX/PlatformDarwinKernel.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteAppleTV.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteAppleWatch.h"
#include "Plugins/Platform/MacOSX/PlatformiOSSimulator.h"
#include "Plugins/Process/MacOSX-Kernel/ProcessKDP.h"
#include "Plugins/SymbolVendor/MacOSX/SymbolVendorMacOSX.h"
#endif
#include "Plugins/StructuredData/DarwinLog/StructuredDataDarwinLog.h"

#if defined(__FreeBSD__)
#include "Plugins/Process/FreeBSD/ProcessFreeBSD.h"
#endif

#if defined(_WIN32)
#include "Plugins/Process/Windows/Common/ProcessWindows.h"
#include "lldb/Host/windows/windows.h"
#endif

#include "llvm/Support/TargetSelect.h"

#include <string>

using namespace lldb_private;

#ifndef LLDB_DISABLE_PYTHON

// Defined in the SWIG source file
#if PY_MAJOR_VERSION >= 3
extern "C" PyObject *PyInit__lldb(void);

#define LLDBSwigPyInit PyInit__lldb

#else
extern "C" void init_lldb(void);

#define LLDBSwigPyInit init_lldb
#endif

// these are the Pythonic implementations of the required callbacks these are
// scripting-language specific, which is why they belong here we still need to
// use function pointers to them instead of relying on linkage-time resolution
// because the SWIG stuff and this file get built at different times
extern "C" bool LLDBSwigPythonBreakpointCallbackFunction(
    const char *python_function_name, const char *session_dictionary_name,
    const lldb::StackFrameSP &sb_frame,
    const lldb::BreakpointLocationSP &sb_bp_loc);

extern "C" bool LLDBSwigPythonWatchpointCallbackFunction(
    const char *python_function_name, const char *session_dictionary_name,
    const lldb::StackFrameSP &sb_frame, const lldb::WatchpointSP &sb_wp);

extern "C" bool LLDBSwigPythonCallTypeScript(
    const char *python_function_name, void *session_dictionary,
    const lldb::ValueObjectSP &valobj_sp, void **pyfunct_wrapper,
    const lldb::TypeSummaryOptionsSP &options_sp, std::string &retval);

extern "C" void *
LLDBSwigPythonCreateSyntheticProvider(const char *python_class_name,
                                      const char *session_dictionary_name,
                                      const lldb::ValueObjectSP &valobj_sp);

extern "C" void *
LLDBSwigPythonCreateCommandObject(const char *python_class_name,
                                  const char *session_dictionary_name,
                                  const lldb::DebuggerSP debugger_sp);

extern "C" void *LLDBSwigPythonCreateScriptedThreadPlan(
    const char *python_class_name, const char *session_dictionary_name,
    const lldb::ThreadPlanSP &thread_plan_sp);

extern "C" bool LLDBSWIGPythonCallThreadPlan(void *implementor,
                                             const char *method_name,
                                             Event *event_sp, bool &got_error);

extern "C" size_t LLDBSwigPython_CalculateNumChildren(void *implementor,
                                                      uint32_t max);

extern "C" void *LLDBSwigPython_GetChildAtIndex(void *implementor,
                                                uint32_t idx);

extern "C" int LLDBSwigPython_GetIndexOfChildWithName(void *implementor,
                                                      const char *child_name);

extern "C" void *LLDBSWIGPython_CastPyObjectToSBValue(void *data);

extern lldb::ValueObjectSP
LLDBSWIGPython_GetValueObjectSPFromSBValue(void *data);

extern "C" bool LLDBSwigPython_UpdateSynthProviderInstance(void *implementor);

extern "C" bool
LLDBSwigPython_MightHaveChildrenSynthProviderInstance(void *implementor);

extern "C" void *
LLDBSwigPython_GetValueSynthProviderInstance(void *implementor);

extern "C" bool
LLDBSwigPythonCallCommand(const char *python_function_name,
                          const char *session_dictionary_name,
                          lldb::DebuggerSP &debugger, const char *args,
                          lldb_private::CommandReturnObject &cmd_retobj,
                          lldb::ExecutionContextRefSP exe_ctx_ref_sp);

extern "C" bool
LLDBSwigPythonCallCommandObject(void *implementor, lldb::DebuggerSP &debugger,
                                const char *args,
                                lldb_private::CommandReturnObject &cmd_retobj,
                                lldb::ExecutionContextRefSP exe_ctx_ref_sp);

extern "C" bool
LLDBSwigPythonCallModuleInit(const char *python_module_name,
                             const char *session_dictionary_name,
                             lldb::DebuggerSP &debugger);

extern "C" void *
LLDBSWIGPythonCreateOSPlugin(const char *python_class_name,
                             const char *session_dictionary_name,
                             const lldb::ProcessSP &process_sp);

extern "C" bool LLDBSWIGPythonRunScriptKeywordProcess(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::ProcessSP &process, std::string &output);

extern "C" bool LLDBSWIGPythonRunScriptKeywordThread(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::ThreadSP &thread, std::string &output);

extern "C" bool LLDBSWIGPythonRunScriptKeywordTarget(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::TargetSP &target, std::string &output);

extern "C" bool LLDBSWIGPythonRunScriptKeywordFrame(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::StackFrameSP &frame, std::string &output);

extern "C" bool LLDBSWIGPythonRunScriptKeywordValue(
    const char *python_function_name, const char *session_dictionary_name,
    lldb::ValueObjectSP &value, std::string &output);

extern "C" void *
LLDBSWIGPython_GetDynamicSetting(void *module, const char *setting,
                                 const lldb::TargetSP &target_sp);

#endif

SystemInitializerFull::SystemInitializerFull() {}

SystemInitializerFull::~SystemInitializerFull() {}

void SystemInitializerFull::Initialize() {
  SystemInitializerCommon::Initialize();

  ObjectFileELF::Initialize();
  ObjectFileMachO::Initialize();
  ObjectFilePECOFF::Initialize();

  ScriptInterpreterNone::Initialize();

#ifndef LLDB_DISABLE_PYTHON
  OperatingSystemPython::Initialize();
#endif
  OperatingSystemGo::Initialize();

#if !defined(LLDB_DISABLE_PYTHON)
  InitializeSWIG();

  // ScriptInterpreterPython::Initialize() depends on things like HostInfo
  // being initialized so it can compute the python directory etc, so we need
  // to do this after SystemInitializerCommon::Initialize().
  ScriptInterpreterPython::Initialize();
#endif

  platform_freebsd::PlatformFreeBSD::Initialize();
  platform_linux::PlatformLinux::Initialize();
  platform_netbsd::PlatformNetBSD::Initialize();
  platform_openbsd::PlatformOpenBSD::Initialize();
  PlatformWindows::Initialize();
  PlatformKalimba::Initialize();
  platform_android::PlatformAndroid::Initialize();
  PlatformRemoteiOS::Initialize();
  PlatformMacOSX::Initialize();
#if defined(__APPLE__)
  PlatformiOSSimulator::Initialize();
  PlatformDarwinKernel::Initialize();
#endif

  // Initialize LLVM and Clang
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();

  ClangASTContext::Initialize();
  GoASTContext::Initialize();
  JavaASTContext::Initialize();
  OCamlASTContext::Initialize();

  ABIMacOSX_i386::Initialize();
  ABIMacOSX_arm::Initialize();
  ABIMacOSX_arm64::Initialize();
  ABISysV_arm::Initialize();
  ABISysV_arm64::Initialize();
  ABISysV_hexagon::Initialize();
  ABISysV_i386::Initialize();
  ABISysV_x86_64::Initialize();
  ABISysV_ppc::Initialize();
  ABISysV_ppc64::Initialize();
  ABISysV_mips::Initialize();
  ABISysV_mips64::Initialize();
  ABISysV_s390x::Initialize();

  ArchitectureArm::Initialize();
  ArchitecturePPC64::Initialize();

  DisassemblerLLVMC::Initialize();

  JITLoaderGDB::Initialize();
  ProcessElfCore::Initialize();
  ProcessMachCore::Initialize();
  minidump::ProcessMinidump::Initialize();
  MemoryHistoryASan::Initialize();
  AddressSanitizerRuntime::Initialize();
  ThreadSanitizerRuntime::Initialize();
  UndefinedBehaviorSanitizerRuntime::Initialize();
  MainThreadCheckerRuntime::Initialize();

  SymbolVendorELF::Initialize();
  SymbolFileDWARF::Initialize();
  SymbolFilePDB::Initialize();
  SymbolFileSymtab::Initialize();
  UnwindAssemblyInstEmulation::Initialize();
  UnwindAssembly_x86::Initialize();
  EmulateInstructionARM64::Initialize();
  EmulateInstructionPPC64::Initialize();
  SymbolFileDWARFDebugMap::Initialize();
  ItaniumABILanguageRuntime::Initialize();
  AppleObjCRuntimeV2::Initialize();
  AppleObjCRuntimeV1::Initialize();
  SystemRuntimeMacOSX::Initialize();
  RenderScriptRuntime::Initialize();
  GoLanguageRuntime::Initialize();
  JavaLanguageRuntime::Initialize();

  CPlusPlusLanguage::Initialize();
  GoLanguage::Initialize();
  JavaLanguage::Initialize();
  ObjCLanguage::Initialize();
  ObjCPlusPlusLanguage::Initialize();
  OCamlLanguage::Initialize();

#if defined(_WIN32)
  ProcessWindows::Initialize();
#endif
#if defined(__FreeBSD__)
  ProcessFreeBSD::Initialize();
#endif
#if defined(__APPLE__)
  SymbolVendorMacOSX::Initialize();
  ProcessKDP::Initialize();
  PlatformAppleTVSimulator::Initialize();
  PlatformAppleWatchSimulator::Initialize();
  PlatformRemoteAppleTV::Initialize();
  PlatformRemoteAppleWatch::Initialize();
  DynamicLoaderDarwinKernel::Initialize();
#endif

  // This plugin is valid on any host that talks to a Darwin remote. It
  // shouldn't be limited to __APPLE__.
  StructuredDataDarwinLog::Initialize();

  //----------------------------------------------------------------------
  // Platform agnostic plugins
  //----------------------------------------------------------------------
  platform_gdb_server::PlatformRemoteGDBServer::Initialize();

  process_gdb_remote::ProcessGDBRemote::Initialize();
  DynamicLoaderMacOSXDYLD::Initialize();
  DynamicLoaderMacOS::Initialize();
  DynamicLoaderPOSIXDYLD::Initialize();
  DynamicLoaderStatic::Initialize();
  DynamicLoaderWindowsDYLD::Initialize();

  // Scan for any system or user LLDB plug-ins
  PluginManager::Initialize();

  // The process settings need to know about installed plug-ins, so the
  // Settings must be initialized
  // AFTER PluginManager::Initialize is called.

  Debugger::SettingsInitialize();
}

void SystemInitializerFull::InitializeSWIG() {
#if !defined(LLDB_DISABLE_PYTHON)
  ScriptInterpreterPython::InitializeInterpreter(
      LLDBSwigPyInit, LLDBSwigPythonBreakpointCallbackFunction,
      LLDBSwigPythonWatchpointCallbackFunction, LLDBSwigPythonCallTypeScript,
      LLDBSwigPythonCreateSyntheticProvider, LLDBSwigPythonCreateCommandObject,
      LLDBSwigPython_CalculateNumChildren, LLDBSwigPython_GetChildAtIndex,
      LLDBSwigPython_GetIndexOfChildWithName,
      LLDBSWIGPython_CastPyObjectToSBValue,
      LLDBSWIGPython_GetValueObjectSPFromSBValue,
      LLDBSwigPython_UpdateSynthProviderInstance,
      LLDBSwigPython_MightHaveChildrenSynthProviderInstance,
      LLDBSwigPython_GetValueSynthProviderInstance, LLDBSwigPythonCallCommand,
      LLDBSwigPythonCallCommandObject, LLDBSwigPythonCallModuleInit,
      LLDBSWIGPythonCreateOSPlugin, LLDBSWIGPythonRunScriptKeywordProcess,
      LLDBSWIGPythonRunScriptKeywordThread,
      LLDBSWIGPythonRunScriptKeywordTarget, LLDBSWIGPythonRunScriptKeywordFrame,
      LLDBSWIGPythonRunScriptKeywordValue, LLDBSWIGPython_GetDynamicSetting,
      LLDBSwigPythonCreateScriptedThreadPlan, LLDBSWIGPythonCallThreadPlan);
#endif
}

void SystemInitializerFull::Terminate() {
  static Timer::Category func_cat(LLVM_PRETTY_FUNCTION);
  Timer scoped_timer(func_cat, LLVM_PRETTY_FUNCTION);

  Debugger::SettingsTerminate();

  // Terminate and unload and loaded system or user LLDB plug-ins
  PluginManager::Terminate();

  ClangASTContext::Terminate();
  GoASTContext::Terminate();
  JavaASTContext::Terminate();
  OCamlASTContext::Terminate();

  ABIMacOSX_i386::Terminate();
  ABIMacOSX_arm::Terminate();
  ABIMacOSX_arm64::Terminate();
  ABISysV_arm::Terminate();
  ABISysV_arm64::Terminate();
  ABISysV_hexagon::Terminate();
  ABISysV_i386::Terminate();
  ABISysV_x86_64::Terminate();
  ABISysV_ppc::Terminate();
  ABISysV_ppc64::Terminate();
  ABISysV_mips::Terminate();
  ABISysV_mips64::Terminate();
  ABISysV_s390x::Terminate();
  DisassemblerLLVMC::Terminate();

  JITLoaderGDB::Terminate();
  ProcessElfCore::Terminate();
  ProcessMachCore::Terminate();
  minidump::ProcessMinidump::Terminate();
  MemoryHistoryASan::Terminate();
  AddressSanitizerRuntime::Terminate();
  ThreadSanitizerRuntime::Terminate();
  UndefinedBehaviorSanitizerRuntime::Terminate();
  MainThreadCheckerRuntime::Terminate();
  SymbolVendorELF::Terminate();
  SymbolFileDWARF::Terminate();
  SymbolFilePDB::Terminate();
  SymbolFileSymtab::Terminate();
  UnwindAssembly_x86::Terminate();
  UnwindAssemblyInstEmulation::Terminate();
  EmulateInstructionARM64::Terminate();
  EmulateInstructionPPC64::Terminate();
  SymbolFileDWARFDebugMap::Terminate();
  ItaniumABILanguageRuntime::Terminate();
  AppleObjCRuntimeV2::Terminate();
  AppleObjCRuntimeV1::Terminate();
  SystemRuntimeMacOSX::Terminate();
  RenderScriptRuntime::Terminate();
  JavaLanguageRuntime::Terminate();

  CPlusPlusLanguage::Terminate();
  GoLanguage::Terminate();
  JavaLanguage::Terminate();
  ObjCLanguage::Terminate();
  ObjCPlusPlusLanguage::Terminate();
  OCamlLanguage::Terminate();

#if defined(__APPLE__)
  DynamicLoaderDarwinKernel::Terminate();
  ProcessKDP::Terminate();
  SymbolVendorMacOSX::Terminate();
  PlatformAppleTVSimulator::Terminate();
  PlatformAppleWatchSimulator::Terminate();
  PlatformRemoteAppleTV::Terminate();
  PlatformRemoteAppleWatch::Terminate();
#endif

#if defined(__FreeBSD__)
  ProcessFreeBSD::Terminate();
#endif
  Debugger::SettingsTerminate();

  platform_gdb_server::PlatformRemoteGDBServer::Terminate();
  process_gdb_remote::ProcessGDBRemote::Terminate();
  StructuredDataDarwinLog::Terminate();

  DynamicLoaderMacOSXDYLD::Terminate();
  DynamicLoaderMacOS::Terminate();
  DynamicLoaderPOSIXDYLD::Terminate();
  DynamicLoaderStatic::Terminate();
  DynamicLoaderWindowsDYLD::Terminate();

#ifndef LLDB_DISABLE_PYTHON
  OperatingSystemPython::Terminate();
#endif
  OperatingSystemGo::Terminate();

  platform_freebsd::PlatformFreeBSD::Terminate();
  platform_linux::PlatformLinux::Terminate();
  platform_netbsd::PlatformNetBSD::Terminate();
  platform_openbsd::PlatformOpenBSD::Terminate();
  PlatformWindows::Terminate();
  PlatformKalimba::Terminate();
  platform_android::PlatformAndroid::Terminate();
  PlatformMacOSX::Terminate();
  PlatformRemoteiOS::Terminate();
#if defined(__APPLE__)
  PlatformiOSSimulator::Terminate();
  PlatformDarwinKernel::Terminate();
#endif

  ObjectFileELF::Terminate();
  ObjectFileMachO::Terminate();
  ObjectFilePECOFF::Terminate();

  // Now shutdown the common parts, in reverse order.
  SystemInitializerCommon::Terminate();
}
