//===-- lldb-private-interfaces.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_lldb_private_interfaces_h_
#define liblldb_lldb_private_interfaces_h_

#if defined(__cplusplus)

#include "lldb/lldb-private.h"

namespace lldb_private
{
    typedef lldb::ABISP (*ABICreateInstance) (const ArchSpec &arch);
    typedef Disassembler* (*DisassemblerCreateInstance) (const ArchSpec &arch, const char *flavor);
    typedef DynamicLoader* (*DynamicLoaderCreateInstance) (Process* process, bool force);
    typedef ObjectContainer* (*ObjectContainerCreateInstance) (const lldb::ModuleSP &module_sp, lldb::DataBufferSP& data_sp, lldb::offset_t data_offset, const FileSpec *file, lldb::offset_t offset, lldb::offset_t length);
    typedef size_t (*ObjectFileGetModuleSpecifications) (const FileSpec &file, lldb::DataBufferSP& data_sp, lldb::offset_t data_offset, lldb::offset_t file_offset, lldb::offset_t length, ModuleSpecList &module_specs);
    typedef ObjectFile* (*ObjectFileCreateInstance) (const lldb::ModuleSP &module_sp, lldb::DataBufferSP& data_sp, lldb::offset_t data_offset, const FileSpec* file, lldb::offset_t file_offset, lldb::offset_t length);
    typedef ObjectFile* (*ObjectFileCreateMemoryInstance) (const lldb::ModuleSP &module_sp, lldb::DataBufferSP& data_sp, const lldb::ProcessSP &process_sp, lldb::addr_t offset);
    typedef LogChannel* (*LogChannelCreateInstance) ();
    typedef EmulateInstruction * (*EmulateInstructionCreateInstance) (const ArchSpec &arch, InstructionType inst_type);
    typedef OperatingSystem* (*OperatingSystemCreateInstance) (Process *process, bool force);
    typedef LanguageRuntime *(*LanguageRuntimeCreateInstance) (Process *process, lldb::LanguageType language);
    typedef Platform* (*PlatformCreateInstance) (bool force, const ArchSpec *arch);
    typedef lldb::ProcessSP (*ProcessCreateInstance) (Target &target, Listener &listener, const FileSpec *crash_file_path);
    typedef SymbolFile* (*SymbolFileCreateInstance) (ObjectFile* obj_file);
    typedef SymbolVendor* (*SymbolVendorCreateInstance) (const lldb::ModuleSP &module_sp, lldb_private::Stream *feedback_strm);   // Module can be NULL for default system symbol vendor
    typedef bool (*BreakpointHitCallback) (void *baton, StoppointCallbackContext *context, lldb::user_id_t break_id, lldb::user_id_t break_loc_id);
    typedef bool (*WatchpointHitCallback) (void *baton, StoppointCallbackContext *context, lldb::user_id_t watch_id);
    typedef ThreadPlan * (*ThreadPlanShouldStopHereCallback) (ThreadPlan *current_plan, Flags &flags, void *baton);
    typedef UnwindAssembly* (*UnwindAssemblyCreateInstance) (const ArchSpec &arch);
    typedef int (*ComparisonFunction)(const void *, const void *);
    typedef bool (*CommandOverrideCallback)(void *baton, const char **argv);
    typedef void (*DebuggerInitializeCallback)(Debugger &debugger);

} // namespace lldb_private

#endif  // #if defined(__cplusplus)

#endif  // liblldb_lldb_private_interfaces_h_
