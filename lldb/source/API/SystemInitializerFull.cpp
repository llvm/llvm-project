//===-- SystemInitializerFull.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemInitializerFull.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/Host/Config.h"

#if LLDB_ENABLE_PYTHON
#include "Plugins/ScriptInterpreter/Python/ScriptInterpreterPython.h"
#endif

#if LLDB_ENABLE_LUA
#include "Plugins/ScriptInterpreter/Lua/ScriptInterpreterLua.h"
#endif

#include "lldb/Core/Debugger.h"
#include "lldb/Host/Host.h"
#include "lldb/Initialization/SystemInitializerCommon.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Symbol/TypeSystemClang.h"
#include "lldb/Utility/Timer.h"

#include "Plugins/ABI/MacOSX-arm/ABIMacOSX_arm.h"
#include "Plugins/ABI/MacOSX-arm64/ABIMacOSX_arm64.h"
#include "Plugins/ABI/MacOSX-i386/ABIMacOSX_i386.h"
#include "Plugins/ABI/SysV-arc/ABISysV_arc.h"
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
#include "Plugins/ABI/Windows-x86_64/ABIWindows_x86_64.h"
#include "Plugins/Architecture/Arm/ArchitectureArm.h"
#include "Plugins/Architecture/Mips/ArchitectureMips.h"
#include "Plugins/Architecture/PPC64/ArchitecturePPC64.h"
#include "Plugins/Disassembler/LLVMC/DisassemblerLLVMC.h"
#include "Plugins/DynamicLoader/MacOSX-DYLD/DynamicLoaderMacOS.h"
#include "Plugins/DynamicLoader/MacOSX-DYLD/DynamicLoaderMacOSXDYLD.h"
#include "Plugins/DynamicLoader/POSIX-DYLD/DynamicLoaderPOSIXDYLD.h"
#include "Plugins/DynamicLoader/Static/DynamicLoaderStatic.h"
#include "Plugins/DynamicLoader/Windows-DYLD/DynamicLoaderWindowsDYLD.h"
#include "Plugins/Instruction/ARM/EmulateInstructionARM.h"
#include "Plugins/Instruction/ARM64/EmulateInstructionARM64.h"
#include "Plugins/Instruction/MIPS/EmulateInstructionMIPS.h"
#include "Plugins/Instruction/MIPS64/EmulateInstructionMIPS64.h"
#include "Plugins/Instruction/PPC64/EmulateInstructionPPC64.h"
#include "Plugins/InstrumentationRuntime/ASan/InstrumentationRuntimeASan.h"
#include "Plugins/InstrumentationRuntime/MainThreadChecker/InstrumentationRuntimeMainThreadChecker.h"
#include "Plugins/InstrumentationRuntime/TSan/InstrumentationRuntimeTSan.h"
#include "Plugins/InstrumentationRuntime/UBSan/InstrumentationRuntimeUBSan.h"
#include "Plugins/JITLoader/GDB/JITLoaderGDB.h"
#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"
#include "Plugins/Language/ObjC/ObjCLanguage.h"
#include "Plugins/Language/ObjCPlusPlus/ObjCPlusPlusLanguage.h"
#include "Plugins/LanguageRuntime/CPlusPlus/ItaniumABI/ItaniumABILanguageRuntime.h"
#include "Plugins/LanguageRuntime/ObjC/AppleObjCRuntime/AppleObjCRuntime.h"
#include "Plugins/LanguageRuntime/RenderScript/RenderScriptRuntime/RenderScriptRuntime.h"
#include "Plugins/MemoryHistory/asan/MemoryHistoryASan.h"
#include "Plugins/ObjectContainer/BSD-Archive/ObjectContainerBSDArchive.h"
#include "Plugins/ObjectContainer/Universal-Mach-O/ObjectContainerUniversalMachO.h"
#include "Plugins/ObjectFile/Breakpad/ObjectFileBreakpad.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "Plugins/ObjectFile/wasm/ObjectFileWasm.h"
#include "Plugins/OperatingSystem/Python/OperatingSystemPython.h"
#include "Plugins/Platform/Android/PlatformAndroid.h"
#include "Plugins/Platform/FreeBSD/PlatformFreeBSD.h"
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
#include "Plugins/SymbolFile/Breakpad/SymbolFileBreakpad.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARFDebugMap.h"
#include "Plugins/SymbolFile/PDB/SymbolFilePDB.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "Plugins/SymbolVendor/ELF/SymbolVendorELF.h"
#include "Plugins/SymbolVendor/wasm/SymbolVendorWasm.h"
#include "Plugins/SystemRuntime/MacOSX/SystemRuntimeMacOSX.h"
#include "Plugins/UnwindAssembly/InstEmulation/UnwindAssemblyInstEmulation.h"
#include "Plugins/UnwindAssembly/x86/UnwindAssembly-x86.h"

#if defined(__APPLE__)
#include "Plugins/DynamicLoader/Darwin-Kernel/DynamicLoaderDarwinKernel.h"
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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#include "llvm/ExecutionEngine/MCJIT.h"
#pragma clang diagnostic pop

#include <string>

using namespace lldb_private;

SystemInitializerFull::SystemInitializerFull() {}

SystemInitializerFull::~SystemInitializerFull() {}

#define LLDB_PROCESS_AArch64(op)                                               \
  ABIMacOSX_arm64::op();                                                       \
  ABISysV_arm64::op();
#define LLDB_PROCESS_ARM(op)                                                   \
  ABIMacOSX_arm::op();                                                         \
  ABISysV_arm::op();
#define LLDB_PROCESS_ARC(op)                                                   \
  ABISysV_arc::op();
#define LLDB_PROCESS_Hexagon(op) ABISysV_hexagon::op();
#define LLDB_PROCESS_Mips(op)                                                  \
  ABISysV_mips::op();                                                          \
  ABISysV_mips64::op();
#define LLDB_PROCESS_PowerPC(op)                                               \
  ABISysV_ppc::op();                                                          \
  ABISysV_ppc64::op();
#define LLDB_PROCESS_SystemZ(op) ABISysV_s390x::op();
#define LLDB_PROCESS_X86(op)                                                   \
  ABIMacOSX_i386::op();                                                        \
  ABISysV_i386::op();                                                          \
  ABISysV_x86_64::op();                                                        \
  ABIWindows_x86_64::op();

#define LLDB_PROCESS_AMDGPU(op)
#define LLDB_PROCESS_AVR(op)
#define LLDB_PROCESS_BPF(op)
#define LLDB_PROCESS_Lanai(op)
#define LLDB_PROCESS_MSP430(op)
#define LLDB_PROCESS_NVPTX(op)
#define LLDB_PROCESS_RISCV(op)
#define LLDB_PROCESS_Sparc(op)
#define LLDB_PROCESS_WebAssembly(op)
#define LLDB_PROCESS_XCore(op)

llvm::Error SystemInitializerFull::Initialize() {
  if (auto e = SystemInitializerCommon::Initialize())
    return e;

  breakpad::ObjectFileBreakpad::Initialize();
  ObjectFileELF::Initialize();
  ObjectFileMachO::Initialize();
  ObjectFilePECOFF::Initialize();
  wasm::ObjectFileWasm::Initialize();

  ObjectContainerBSDArchive::Initialize();
  ObjectContainerUniversalMachO::Initialize();

  ScriptInterpreterNone::Initialize();

#if LLDB_ENABLE_PYTHON
  OperatingSystemPython::Initialize();
#endif

#if LLDB_ENABLE_PYTHON
  ScriptInterpreterPython::Initialize();
#endif

#if LLDB_ENABLE_LUA
  ScriptInterpreterLua::Initialize();
#endif

  platform_freebsd::PlatformFreeBSD::Initialize();
  platform_linux::PlatformLinux::Initialize();
  platform_netbsd::PlatformNetBSD::Initialize();
  platform_openbsd::PlatformOpenBSD::Initialize();
  PlatformWindows::Initialize();
  platform_android::PlatformAndroid::Initialize();
  PlatformRemoteiOS::Initialize();
  PlatformMacOSX::Initialize();

  // Initialize LLVM and Clang
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();

  TypeSystemClang::Initialize();

#define LLVM_TARGET(t) LLDB_PROCESS_ ## t(Initialize)
#include "llvm/Config/Targets.def"

  ArchitectureArm::Initialize();
  ArchitectureMips::Initialize();
  ArchitecturePPC64::Initialize();

  DisassemblerLLVMC::Initialize();

  JITLoaderGDB::Initialize();
  ProcessElfCore::Initialize();
  ProcessMachCore::Initialize();
  minidump::ProcessMinidump::Initialize();
  MemoryHistoryASan::Initialize();
  InstrumentationRuntimeASan::Initialize();
  InstrumentationRuntimeTSan::Initialize();
  InstrumentationRuntimeUBSan::Initialize();
  InstrumentationRuntimeMainThreadChecker::Initialize();

  SymbolVendorELF::Initialize();
  breakpad::SymbolFileBreakpad::Initialize();
  SymbolFileDWARF::Initialize();
  SymbolFilePDB::Initialize();
  SymbolFileSymtab::Initialize();
  wasm::SymbolVendorWasm::Initialize();
  UnwindAssemblyInstEmulation::Initialize();
  UnwindAssembly_x86::Initialize();

  EmulateInstructionARM::Initialize();
  EmulateInstructionARM64::Initialize();
  EmulateInstructionMIPS::Initialize();
  EmulateInstructionMIPS64::Initialize();
  EmulateInstructionPPC64::Initialize();

  SymbolFileDWARFDebugMap::Initialize();
  ItaniumABILanguageRuntime::Initialize();
  AppleObjCRuntime::Initialize();
  SystemRuntimeMacOSX::Initialize();
  RenderScriptRuntime::Initialize();

  CPlusPlusLanguage::Initialize();
  ObjCLanguage::Initialize();
  ObjCPlusPlusLanguage::Initialize();

#if defined(_WIN32)
  ProcessWindows::Initialize();
#endif
#if defined(__FreeBSD__)
  ProcessFreeBSD::Initialize();
#endif
#if defined(__APPLE__)
  SymbolVendorMacOSX::Initialize();
  ProcessKDP::Initialize();
  DynamicLoaderDarwinKernel::Initialize();
#endif

  // This plugin is valid on any host that talks to a Darwin remote. It
  // shouldn't be limited to __APPLE__.
  StructuredDataDarwinLog::Initialize();

  // Platform agnostic plugins
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

  return llvm::Error::success();
}

void SystemInitializerFull::Terminate() {
  static Timer::Category func_cat(LLVM_PRETTY_FUNCTION);
  Timer scoped_timer(func_cat, LLVM_PRETTY_FUNCTION);

  Debugger::SettingsTerminate();

  // Terminate and unload and loaded system or user LLDB plug-ins
  PluginManager::Terminate();

  TypeSystemClang::Terminate();

  ArchitectureArm::Terminate();
  ArchitectureMips::Terminate();
  ArchitecturePPC64::Terminate();

#define LLVM_TARGET(t) LLDB_PROCESS_ ## t(Terminate)
#include "llvm/Config/Targets.def"

  DisassemblerLLVMC::Terminate();

  JITLoaderGDB::Terminate();
  ProcessElfCore::Terminate();
  ProcessMachCore::Terminate();
  minidump::ProcessMinidump::Terminate();
  MemoryHistoryASan::Terminate();
  InstrumentationRuntimeASan::Terminate();
  InstrumentationRuntimeTSan::Terminate();
  InstrumentationRuntimeUBSan::Terminate();
  InstrumentationRuntimeMainThreadChecker::Terminate();

  wasm::SymbolVendorWasm::Terminate();
  SymbolVendorELF::Terminate();
  breakpad::SymbolFileBreakpad::Terminate();
  SymbolFileDWARF::Terminate();
  SymbolFilePDB::Terminate();
  SymbolFileSymtab::Terminate();
  UnwindAssembly_x86::Terminate();
  UnwindAssemblyInstEmulation::Terminate();

  EmulateInstructionARM::Terminate();
  EmulateInstructionARM64::Terminate();
  EmulateInstructionMIPS::Terminate();
  EmulateInstructionMIPS64::Terminate();
  EmulateInstructionPPC64::Terminate();

  SymbolFileDWARFDebugMap::Terminate();
  ItaniumABILanguageRuntime::Terminate();
  AppleObjCRuntime::Terminate();
  SystemRuntimeMacOSX::Terminate();
  RenderScriptRuntime::Terminate();

  CPlusPlusLanguage::Terminate();
  ObjCLanguage::Terminate();
  ObjCPlusPlusLanguage::Terminate();

#if defined(__APPLE__)
  DynamicLoaderDarwinKernel::Terminate();
  ProcessKDP::Terminate();
  SymbolVendorMacOSX::Terminate();
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

  platform_freebsd::PlatformFreeBSD::Terminate();
  platform_linux::PlatformLinux::Terminate();
  platform_netbsd::PlatformNetBSD::Terminate();
  platform_openbsd::PlatformOpenBSD::Terminate();
  PlatformWindows::Terminate();
  platform_android::PlatformAndroid::Terminate();
  PlatformMacOSX::Terminate();
  PlatformRemoteiOS::Terminate();

  breakpad::ObjectFileBreakpad::Terminate();
  ObjectFileELF::Terminate();
  ObjectFileMachO::Terminate();
  ObjectFilePECOFF::Terminate();
  wasm::ObjectFileWasm::Terminate();

  ObjectContainerBSDArchive::Terminate();
  ObjectContainerUniversalMachO::Terminate();

#if LLDB_ENABLE_PYTHON
  OperatingSystemPython::Terminate();
#endif

#if LLDB_ENABLE_PYTHON
  ScriptInterpreterPython::Terminate();
#endif

#if LLDB_ENABLE_LUA
  ScriptInterpreterLua::Terminate();
#endif

  // Now shutdown the common parts, in reverse order.
  SystemInitializerCommon::Terminate();
}
