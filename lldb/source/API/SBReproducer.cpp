//===-- SBReproducer.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SBReproducerPrivate.h"

#include "SBReproducerPrivate.h"
#include "lldb/API/LLDB.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBAttachInfo.h"
#include "lldb/API/SBBlock.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBData.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBDeclaration.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBHostOS.h"
#include "lldb/API/SBReproducer.h"

#include "lldb/Host/FileSystem.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::repro;

static void SetFileHandleRedirect(SBDebugger *, FILE *, bool) {
  // Do nothing.
}

static bool GetDefaultArchitectureRedirect(char *arch_name,
                                           size_t arch_name_len) {
  // The function is writing to its argument. Without the redirect it would
  // write into the replay buffer.
  char buffer[arch_name_len];
  return SBDebugger::GetDefaultArchitecture(buffer, arch_name_len);
}

SBRegistry::SBRegistry() {

  // Custom implementation.
  Register(&invoke<void (SBDebugger::*)(
               FILE *, bool)>::method<&SBDebugger::SetErrorFileHandle>::doit,
           &SetFileHandleRedirect);
  Register(&invoke<void (SBDebugger::*)(
               FILE *, bool)>::method<&SBDebugger::SetOutputFileHandle>::doit,
           &SetFileHandleRedirect);
  Register<bool(char *, size_t)>(static_cast<bool (*)(char *, size_t)>(
                                     &SBDebugger::GetDefaultArchitecture),
                                 &GetDefaultArchitectureRedirect);

  {
    LLDB_REGISTER_CONSTRUCTOR(SBAddress, ());
    LLDB_REGISTER_CONSTRUCTOR(SBAddress, (const lldb::SBAddress &));
    LLDB_REGISTER_CONSTRUCTOR(SBAddress, (lldb::SBSection, lldb::addr_t));
    LLDB_REGISTER_CONSTRUCTOR(SBAddress, (lldb::addr_t, lldb::SBTarget &));
    LLDB_REGISTER_METHOD(const lldb::SBAddress &,
                         SBAddress, operator=,(const lldb::SBAddress &));
    LLDB_REGISTER_METHOD_CONST(bool, SBAddress, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBAddress, Clear, ());
    LLDB_REGISTER_METHOD(void, SBAddress, SetAddress,
                         (lldb::SBSection, lldb::addr_t));
    LLDB_REGISTER_METHOD_CONST(lldb::addr_t, SBAddress, GetFileAddress, ());
    LLDB_REGISTER_METHOD_CONST(lldb::addr_t, SBAddress, GetLoadAddress,
                               (const lldb::SBTarget &));
    LLDB_REGISTER_METHOD(void, SBAddress, SetLoadAddress,
                         (lldb::addr_t, lldb::SBTarget &));
    LLDB_REGISTER_METHOD(bool, SBAddress, OffsetAddress, (lldb::addr_t));
    LLDB_REGISTER_METHOD(lldb::SBSection, SBAddress, GetSection, ());
    LLDB_REGISTER_METHOD(lldb::addr_t, SBAddress, GetOffset, ());
    LLDB_REGISTER_METHOD(bool, SBAddress, GetDescription, (lldb::SBStream &));
    LLDB_REGISTER_METHOD(lldb::SBModule, SBAddress, GetModule, ());
    LLDB_REGISTER_METHOD(lldb::SBSymbolContext, SBAddress, GetSymbolContext,
                         (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBCompileUnit, SBAddress, GetCompileUnit, ());
    LLDB_REGISTER_METHOD(lldb::SBFunction, SBAddress, GetFunction, ());
    LLDB_REGISTER_METHOD(lldb::SBBlock, SBAddress, GetBlock, ());
    LLDB_REGISTER_METHOD(lldb::SBSymbol, SBAddress, GetSymbol, ());
    LLDB_REGISTER_METHOD(lldb::SBLineEntry, SBAddress, GetLineEntry, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBAttachInfo, ());
    LLDB_REGISTER_CONSTRUCTOR(SBAttachInfo, (lldb::pid_t));
    LLDB_REGISTER_CONSTRUCTOR(SBAttachInfo, (const char *, bool));
    LLDB_REGISTER_CONSTRUCTOR(SBAttachInfo, (const char *, bool, bool));
    LLDB_REGISTER_CONSTRUCTOR(SBAttachInfo, (const lldb::SBAttachInfo &));
    LLDB_REGISTER_METHOD(lldb::SBAttachInfo &,
                         SBAttachInfo, operator=,(const lldb::SBAttachInfo &));
    LLDB_REGISTER_METHOD(lldb::pid_t, SBAttachInfo, GetProcessID, ());
    LLDB_REGISTER_METHOD(void, SBAttachInfo, SetProcessID, (lldb::pid_t));
    LLDB_REGISTER_METHOD(uint32_t, SBAttachInfo, GetResumeCount, ());
    LLDB_REGISTER_METHOD(void, SBAttachInfo, SetResumeCount, (uint32_t));
    LLDB_REGISTER_METHOD(const char *, SBAttachInfo, GetProcessPluginName, ());
    LLDB_REGISTER_METHOD(void, SBAttachInfo, SetProcessPluginName,
                         (const char *));
    LLDB_REGISTER_METHOD(void, SBAttachInfo, SetExecutable, (const char *));
    LLDB_REGISTER_METHOD(void, SBAttachInfo, SetExecutable, (lldb::SBFileSpec));
    LLDB_REGISTER_METHOD(bool, SBAttachInfo, GetWaitForLaunch, ());
    LLDB_REGISTER_METHOD(void, SBAttachInfo, SetWaitForLaunch, (bool));
    LLDB_REGISTER_METHOD(void, SBAttachInfo, SetWaitForLaunch, (bool, bool));
    LLDB_REGISTER_METHOD(bool, SBAttachInfo, GetIgnoreExisting, ());
    LLDB_REGISTER_METHOD(void, SBAttachInfo, SetIgnoreExisting, (bool));
    LLDB_REGISTER_METHOD(uint32_t, SBAttachInfo, GetUserID, ());
    LLDB_REGISTER_METHOD(uint32_t, SBAttachInfo, GetGroupID, ());
    LLDB_REGISTER_METHOD(bool, SBAttachInfo, UserIDIsValid, ());
    LLDB_REGISTER_METHOD(bool, SBAttachInfo, GroupIDIsValid, ());
    LLDB_REGISTER_METHOD(void, SBAttachInfo, SetUserID, (uint32_t));
    LLDB_REGISTER_METHOD(void, SBAttachInfo, SetGroupID, (uint32_t));
    LLDB_REGISTER_METHOD(uint32_t, SBAttachInfo, GetEffectiveUserID, ());
    LLDB_REGISTER_METHOD(uint32_t, SBAttachInfo, GetEffectiveGroupID, ());
    LLDB_REGISTER_METHOD(bool, SBAttachInfo, EffectiveUserIDIsValid, ());
    LLDB_REGISTER_METHOD(bool, SBAttachInfo, EffectiveGroupIDIsValid, ());
    LLDB_REGISTER_METHOD(void, SBAttachInfo, SetEffectiveUserID, (uint32_t));
    LLDB_REGISTER_METHOD(void, SBAttachInfo, SetEffectiveGroupID, (uint32_t));
    LLDB_REGISTER_METHOD(lldb::pid_t, SBAttachInfo, GetParentProcessID, ());
    LLDB_REGISTER_METHOD(void, SBAttachInfo, SetParentProcessID, (lldb::pid_t));
    LLDB_REGISTER_METHOD(bool, SBAttachInfo, ParentProcessIDIsValid, ());
    LLDB_REGISTER_METHOD(lldb::SBListener, SBAttachInfo, GetListener, ());
    LLDB_REGISTER_METHOD(void, SBAttachInfo, SetListener, (lldb::SBListener &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBBlock, ());
    LLDB_REGISTER_CONSTRUCTOR(SBBlock, (const lldb::SBBlock &));
    LLDB_REGISTER_METHOD(const lldb::SBBlock &,
                         SBBlock, operator=,(const lldb::SBBlock &));
    LLDB_REGISTER_METHOD_CONST(bool, SBBlock, IsValid, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBBlock, IsInlined, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBBlock, GetInlinedName, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBFileSpec, SBBlock,
                               GetInlinedCallSiteFile, ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBBlock, GetInlinedCallSiteLine, ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBBlock, GetInlinedCallSiteColumn, ());
    LLDB_REGISTER_METHOD(lldb::SBBlock, SBBlock, GetParent, ());
    LLDB_REGISTER_METHOD(lldb::SBBlock, SBBlock, GetContainingInlinedBlock, ());
    LLDB_REGISTER_METHOD(lldb::SBBlock, SBBlock, GetSibling, ());
    LLDB_REGISTER_METHOD(lldb::SBBlock, SBBlock, GetFirstChild, ());
    LLDB_REGISTER_METHOD(bool, SBBlock, GetDescription, (lldb::SBStream &));
    LLDB_REGISTER_METHOD(uint32_t, SBBlock, GetNumRanges, ());
    LLDB_REGISTER_METHOD(lldb::SBAddress, SBBlock, GetRangeStartAddress,
                         (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBAddress, SBBlock, GetRangeEndAddress,
                         (uint32_t));
    LLDB_REGISTER_METHOD(uint32_t, SBBlock, GetRangeIndexForBlockAddress,
                         (lldb::SBAddress));
    LLDB_REGISTER_METHOD(
        lldb::SBValueList, SBBlock, GetVariables,
        (lldb::SBFrame &, bool, bool, bool, lldb::DynamicValueType));
    LLDB_REGISTER_METHOD(lldb::SBValueList, SBBlock, GetVariables,
                         (lldb::SBTarget &, bool, bool, bool));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBBreakpoint, ());
    LLDB_REGISTER_CONSTRUCTOR(SBBreakpoint, (const lldb::SBBreakpoint &));
    LLDB_REGISTER_CONSTRUCTOR(SBBreakpoint, (const lldb::BreakpointSP &));
    LLDB_REGISTER_METHOD(const lldb::SBBreakpoint &,
                         SBBreakpoint, operator=,(const lldb::SBBreakpoint &));
    LLDB_REGISTER_METHOD(bool,
                         SBBreakpoint, operator==,(const lldb::SBBreakpoint &));
    LLDB_REGISTER_METHOD(bool,
                         SBBreakpoint, operator!=,(const lldb::SBBreakpoint &));
    LLDB_REGISTER_METHOD_CONST(lldb::break_id_t, SBBreakpoint, GetID, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBBreakpoint, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBBreakpoint, ClearAllBreakpointSites, ());
    LLDB_REGISTER_METHOD(lldb::SBBreakpointLocation, SBBreakpoint,
                         FindLocationByAddress, (lldb::addr_t));
    LLDB_REGISTER_METHOD(lldb::break_id_t, SBBreakpoint,
                         FindLocationIDByAddress, (lldb::addr_t));
    LLDB_REGISTER_METHOD(lldb::SBBreakpointLocation, SBBreakpoint,
                         FindLocationByID, (lldb::break_id_t));
    LLDB_REGISTER_METHOD(lldb::SBBreakpointLocation, SBBreakpoint,
                         GetLocationAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(void, SBBreakpoint, SetEnabled, (bool));
    LLDB_REGISTER_METHOD(bool, SBBreakpoint, IsEnabled, ());
    LLDB_REGISTER_METHOD(void, SBBreakpoint, SetOneShot, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBBreakpoint, IsOneShot, ());
    LLDB_REGISTER_METHOD(bool, SBBreakpoint, IsInternal, ());
    LLDB_REGISTER_METHOD(void, SBBreakpoint, SetIgnoreCount, (uint32_t));
    LLDB_REGISTER_METHOD(void, SBBreakpoint, SetCondition, (const char *));
    LLDB_REGISTER_METHOD(const char *, SBBreakpoint, GetCondition, ());
    LLDB_REGISTER_METHOD(void, SBBreakpoint, SetAutoContinue, (bool));
    LLDB_REGISTER_METHOD(bool, SBBreakpoint, GetAutoContinue, ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBBreakpoint, GetHitCount, ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBBreakpoint, GetIgnoreCount, ());
    LLDB_REGISTER_METHOD(void, SBBreakpoint, SetThreadID, (lldb::tid_t));
    LLDB_REGISTER_METHOD(lldb::tid_t, SBBreakpoint, GetThreadID, ());
    LLDB_REGISTER_METHOD(void, SBBreakpoint, SetThreadIndex, (uint32_t));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBBreakpoint, GetThreadIndex, ());
    LLDB_REGISTER_METHOD(void, SBBreakpoint, SetThreadName, (const char *));
    LLDB_REGISTER_METHOD_CONST(const char *, SBBreakpoint, GetThreadName, ());
    LLDB_REGISTER_METHOD(void, SBBreakpoint, SetQueueName, (const char *));
    LLDB_REGISTER_METHOD_CONST(const char *, SBBreakpoint, GetQueueName, ());
    LLDB_REGISTER_METHOD_CONST(size_t, SBBreakpoint, GetNumResolvedLocations,
                               ());
    LLDB_REGISTER_METHOD_CONST(size_t, SBBreakpoint, GetNumLocations, ());
    LLDB_REGISTER_METHOD(void, SBBreakpoint, SetCommandLineCommands,
                         (lldb::SBStringList &));
    LLDB_REGISTER_METHOD(bool, SBBreakpoint, GetCommandLineCommands,
                         (lldb::SBStringList &));
    LLDB_REGISTER_METHOD(bool, SBBreakpoint, GetDescription,
                         (lldb::SBStream &));
    LLDB_REGISTER_METHOD(bool, SBBreakpoint, GetDescription,
                         (lldb::SBStream &, bool));
    LLDB_REGISTER_METHOD(lldb::SBError, SBBreakpoint, AddLocation,
                         (lldb::SBAddress &));
    LLDB_REGISTER_METHOD(void, SBBreakpoint, SetScriptCallbackFunction,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::SBError, SBBreakpoint, SetScriptCallbackBody,
                         (const char *));
    LLDB_REGISTER_METHOD(bool, SBBreakpoint, AddName, (const char *));
    LLDB_REGISTER_METHOD(void, SBBreakpoint, RemoveName, (const char *));
    LLDB_REGISTER_METHOD(bool, SBBreakpoint, MatchesName, (const char *));
    LLDB_REGISTER_METHOD(void, SBBreakpoint, GetNames, (lldb::SBStringList &));
    LLDB_REGISTER_STATIC_METHOD(bool, SBBreakpoint, EventIsBreakpointEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(lldb::BreakpointEventType, SBBreakpoint,
                                GetBreakpointEventTypeFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBBreakpoint, SBBreakpoint,
                                GetBreakpointFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBBreakpointLocation, SBBreakpoint,
                                GetBreakpointLocationAtIndexFromEvent,
                                (const lldb::SBEvent &, uint32_t));
    LLDB_REGISTER_STATIC_METHOD(uint32_t, SBBreakpoint,
                                GetNumBreakpointLocationsFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_METHOD_CONST(bool, SBBreakpoint, IsHardware, ());
    LLDB_REGISTER_CONSTRUCTOR(SBBreakpointList, (lldb::SBTarget &));
    LLDB_REGISTER_METHOD_CONST(size_t, SBBreakpointList, GetSize, ());
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBBreakpointList,
                         GetBreakpointAtIndex, (size_t));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBBreakpointList,
                         FindBreakpointByID, (lldb::break_id_t));
    LLDB_REGISTER_METHOD(void, SBBreakpointList, Append,
                         (const lldb::SBBreakpoint &));
    LLDB_REGISTER_METHOD(void, SBBreakpointList, AppendByID,
                         (lldb::break_id_t));
    LLDB_REGISTER_METHOD(bool, SBBreakpointList, AppendIfUnique,
                         (const lldb::SBBreakpoint &));
    LLDB_REGISTER_METHOD(void, SBBreakpointList, Clear, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBBreakpointLocation, ());
    LLDB_REGISTER_CONSTRUCTOR(SBBreakpointLocation,
                              (const lldb::BreakpointLocationSP &));
    LLDB_REGISTER_CONSTRUCTOR(SBBreakpointLocation,
                              (const lldb::SBBreakpointLocation &));
    LLDB_REGISTER_METHOD(
        const lldb::SBBreakpointLocation &,
        SBBreakpointLocation, operator=,(const lldb::SBBreakpointLocation &));
    LLDB_REGISTER_METHOD_CONST(bool, SBBreakpointLocation, IsValid, ());
    LLDB_REGISTER_METHOD(lldb::SBAddress, SBBreakpointLocation, GetAddress, ());
    LLDB_REGISTER_METHOD(lldb::addr_t, SBBreakpointLocation, GetLoadAddress,
                         ());
    LLDB_REGISTER_METHOD(void, SBBreakpointLocation, SetEnabled, (bool));
    LLDB_REGISTER_METHOD(bool, SBBreakpointLocation, IsEnabled, ());
    LLDB_REGISTER_METHOD(uint32_t, SBBreakpointLocation, GetHitCount, ());
    LLDB_REGISTER_METHOD(uint32_t, SBBreakpointLocation, GetIgnoreCount, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointLocation, SetIgnoreCount,
                         (uint32_t));
    LLDB_REGISTER_METHOD(void, SBBreakpointLocation, SetCondition,
                         (const char *));
    LLDB_REGISTER_METHOD(const char *, SBBreakpointLocation, GetCondition, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointLocation, SetAutoContinue, (bool));
    LLDB_REGISTER_METHOD(bool, SBBreakpointLocation, GetAutoContinue, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointLocation, SetScriptCallbackFunction,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::SBError, SBBreakpointLocation,
                         SetScriptCallbackBody, (const char *));
    LLDB_REGISTER_METHOD(void, SBBreakpointLocation, SetCommandLineCommands,
                         (lldb::SBStringList &));
    LLDB_REGISTER_METHOD(bool, SBBreakpointLocation, GetCommandLineCommands,
                         (lldb::SBStringList &));
    LLDB_REGISTER_METHOD(void, SBBreakpointLocation, SetThreadID,
                         (lldb::tid_t));
    LLDB_REGISTER_METHOD(lldb::tid_t, SBBreakpointLocation, GetThreadID, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointLocation, SetThreadIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBBreakpointLocation, GetThreadIndex,
                               ());
    LLDB_REGISTER_METHOD(void, SBBreakpointLocation, SetThreadName,
                         (const char *));
    LLDB_REGISTER_METHOD_CONST(const char *, SBBreakpointLocation,
                               GetThreadName, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointLocation, SetQueueName,
                         (const char *));
    LLDB_REGISTER_METHOD_CONST(const char *, SBBreakpointLocation, GetQueueName,
                               ());
    LLDB_REGISTER_METHOD(bool, SBBreakpointLocation, IsResolved, ());
    LLDB_REGISTER_METHOD(bool, SBBreakpointLocation, GetDescription,
                         (lldb::SBStream &, lldb::DescriptionLevel));
    LLDB_REGISTER_METHOD(lldb::break_id_t, SBBreakpointLocation, GetID, ());
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBBreakpointLocation,
                         GetBreakpoint, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBBreakpointName, ());
    LLDB_REGISTER_CONSTRUCTOR(SBBreakpointName,
                              (lldb::SBTarget &, const char *));
    LLDB_REGISTER_CONSTRUCTOR(SBBreakpointName,
                              (lldb::SBBreakpoint &, const char *));
    LLDB_REGISTER_CONSTRUCTOR(SBBreakpointName,
                              (const lldb::SBBreakpointName &));
    LLDB_REGISTER_METHOD(
        const lldb::SBBreakpointName &,
        SBBreakpointName, operator=,(const lldb::SBBreakpointName &));
    LLDB_REGISTER_METHOD(
        bool, SBBreakpointName, operator==,(const lldb::SBBreakpointName &));
    LLDB_REGISTER_METHOD(
        bool, SBBreakpointName, operator!=,(const lldb::SBBreakpointName &));
    LLDB_REGISTER_METHOD_CONST(bool, SBBreakpointName, IsValid, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBBreakpointName, GetName, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetEnabled, (bool));
    LLDB_REGISTER_METHOD(bool, SBBreakpointName, IsEnabled, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetOneShot, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBBreakpointName, IsOneShot, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetIgnoreCount, (uint32_t));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBBreakpointName, GetIgnoreCount, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetCondition, (const char *));
    LLDB_REGISTER_METHOD(const char *, SBBreakpointName, GetCondition, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetAutoContinue, (bool));
    LLDB_REGISTER_METHOD(bool, SBBreakpointName, GetAutoContinue, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetThreadID, (lldb::tid_t));
    LLDB_REGISTER_METHOD(lldb::tid_t, SBBreakpointName, GetThreadID, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetThreadIndex, (uint32_t));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBBreakpointName, GetThreadIndex, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetThreadName, (const char *));
    LLDB_REGISTER_METHOD_CONST(const char *, SBBreakpointName, GetThreadName,
                               ());
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetQueueName, (const char *));
    LLDB_REGISTER_METHOD_CONST(const char *, SBBreakpointName, GetQueueName,
                               ());
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetCommandLineCommands,
                         (lldb::SBStringList &));
    LLDB_REGISTER_METHOD(bool, SBBreakpointName, GetCommandLineCommands,
                         (lldb::SBStringList &));
    LLDB_REGISTER_METHOD_CONST(const char *, SBBreakpointName, GetHelpString,
                               ());
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetHelpString, (const char *));
    LLDB_REGISTER_METHOD(bool, SBBreakpointName, GetDescription,
                         (lldb::SBStream &));
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetScriptCallbackFunction,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::SBError, SBBreakpointName, SetScriptCallbackBody,
                         (const char *));
    LLDB_REGISTER_METHOD_CONST(bool, SBBreakpointName, GetAllowList, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetAllowList, (bool));
    LLDB_REGISTER_METHOD(bool, SBBreakpointName, GetAllowDelete, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetAllowDelete, (bool));
    LLDB_REGISTER_METHOD(bool, SBBreakpointName, GetAllowDisable, ());
    LLDB_REGISTER_METHOD(void, SBBreakpointName, SetAllowDisable, (bool));
  }
  {} {
    LLDB_REGISTER_CONSTRUCTOR(SBBroadcaster, ());
    LLDB_REGISTER_CONSTRUCTOR(SBBroadcaster, (const char *));
    LLDB_REGISTER_CONSTRUCTOR(SBBroadcaster, (const lldb::SBBroadcaster &));
    LLDB_REGISTER_METHOD(
        const lldb::SBBroadcaster &,
        SBBroadcaster, operator=,(const lldb::SBBroadcaster &));
    LLDB_REGISTER_METHOD(void, SBBroadcaster, BroadcastEventByType,
                         (uint32_t, bool));
    LLDB_REGISTER_METHOD(void, SBBroadcaster, BroadcastEvent,
                         (const lldb::SBEvent &, bool));
    LLDB_REGISTER_METHOD(void, SBBroadcaster, AddInitialEventsToListener,
                         (const lldb::SBListener &, uint32_t));
    LLDB_REGISTER_METHOD(uint32_t, SBBroadcaster, AddListener,
                         (const lldb::SBListener &, uint32_t));
    LLDB_REGISTER_METHOD_CONST(const char *, SBBroadcaster, GetName, ());
    LLDB_REGISTER_METHOD(bool, SBBroadcaster, EventTypeHasListeners,
                         (uint32_t));
    LLDB_REGISTER_METHOD(bool, SBBroadcaster, RemoveListener,
                         (const lldb::SBListener &, uint32_t));
    LLDB_REGISTER_METHOD_CONST(bool, SBBroadcaster, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBBroadcaster, Clear, ());
    LLDB_REGISTER_METHOD_CONST(
        bool, SBBroadcaster, operator==,(const lldb::SBBroadcaster &));
    LLDB_REGISTER_METHOD_CONST(
        bool, SBBroadcaster, operator!=,(const lldb::SBBroadcaster &));
    LLDB_REGISTER_METHOD_CONST(
        bool, SBBroadcaster, operator<,(const lldb::SBBroadcaster &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBCommandInterpreterRunOptions, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBCommandInterpreterRunOptions,
                               GetStopOnContinue, ());
    LLDB_REGISTER_METHOD(void, SBCommandInterpreterRunOptions,
                         SetStopOnContinue, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBCommandInterpreterRunOptions,
                               GetStopOnError, ());
    LLDB_REGISTER_METHOD(void, SBCommandInterpreterRunOptions, SetStopOnError,
                         (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBCommandInterpreterRunOptions,
                               GetStopOnCrash, ());
    LLDB_REGISTER_METHOD(void, SBCommandInterpreterRunOptions, SetStopOnCrash,
                         (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBCommandInterpreterRunOptions,
                               GetEchoCommands, ());
    LLDB_REGISTER_METHOD(void, SBCommandInterpreterRunOptions, SetEchoCommands,
                         (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBCommandInterpreterRunOptions,
                               GetEchoCommentCommands, ());
    LLDB_REGISTER_METHOD(void, SBCommandInterpreterRunOptions,
                         SetEchoCommentCommands, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBCommandInterpreterRunOptions,
                               GetPrintResults, ());
    LLDB_REGISTER_METHOD(void, SBCommandInterpreterRunOptions, SetPrintResults,
                         (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBCommandInterpreterRunOptions,
                               GetAddToHistory, ());
    LLDB_REGISTER_METHOD(void, SBCommandInterpreterRunOptions, SetAddToHistory,
                         (bool));
    LLDB_REGISTER_CONSTRUCTOR(SBCommandInterpreter,
                              (lldb_private::CommandInterpreter *));
    LLDB_REGISTER_CONSTRUCTOR(SBCommandInterpreter,
                              (const lldb::SBCommandInterpreter &));
    LLDB_REGISTER_METHOD(
        const lldb::SBCommandInterpreter &,
        SBCommandInterpreter, operator=,(const lldb::SBCommandInterpreter &));
    LLDB_REGISTER_METHOD_CONST(bool, SBCommandInterpreter, IsValid, ());
    LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, CommandExists,
                         (const char *));
    LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, AliasExists,
                         (const char *));
    LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, IsActive, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBCommandInterpreter, WasInterrupted, ());
    LLDB_REGISTER_METHOD(const char *, SBCommandInterpreter,
                         GetIOHandlerControlSequence, (char));
    LLDB_REGISTER_METHOD(lldb::ReturnStatus, SBCommandInterpreter,
                         HandleCommand,
                         (const char *, lldb::SBCommandReturnObject &, bool));
    LLDB_REGISTER_METHOD(lldb::ReturnStatus, SBCommandInterpreter,
                         HandleCommand,
                         (const char *, lldb::SBExecutionContext &,
                          lldb::SBCommandReturnObject &, bool));
    LLDB_REGISTER_METHOD(void, SBCommandInterpreter, HandleCommandsFromFile,
                         (lldb::SBFileSpec &, lldb::SBExecutionContext &,
                          lldb::SBCommandInterpreterRunOptions &,
                          lldb::SBCommandReturnObject));
    LLDB_REGISTER_METHOD(int, SBCommandInterpreter, HandleCompletion,
                         (const char *, const char *, const char *, int, int,
                          lldb::SBStringList &));
    LLDB_REGISTER_METHOD(int, SBCommandInterpreter,
                         HandleCompletionWithDescriptions,
                         (const char *, const char *, const char *, int, int,
                          lldb::SBStringList &, lldb::SBStringList &));
    LLDB_REGISTER_METHOD(int, SBCommandInterpreter,
                         HandleCompletionWithDescriptions,
                         (const char *, uint32_t, int, int,
                          lldb::SBStringList &, lldb::SBStringList &));
    LLDB_REGISTER_METHOD(
        int, SBCommandInterpreter, HandleCompletion,
        (const char *, uint32_t, int, int, lldb::SBStringList &));
    LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, HasCommands, ());
    LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, HasAliases, ());
    LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, HasAliasOptions, ());
    LLDB_REGISTER_METHOD(lldb::SBProcess, SBCommandInterpreter, GetProcess, ());
    LLDB_REGISTER_METHOD(lldb::SBDebugger, SBCommandInterpreter, GetDebugger,
                         ());
    LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, GetPromptOnQuit, ());
    LLDB_REGISTER_METHOD(void, SBCommandInterpreter, SetPromptOnQuit, (bool));
    LLDB_REGISTER_METHOD(void, SBCommandInterpreter, AllowExitCodeOnQuit,
                         (bool));
    LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, HasCustomQuitExitCode, ());
    LLDB_REGISTER_METHOD(int, SBCommandInterpreter, GetQuitStatus, ());
    LLDB_REGISTER_METHOD(void, SBCommandInterpreter, ResolveCommand,
                         (const char *, lldb::SBCommandReturnObject &));
    LLDB_REGISTER_METHOD(void, SBCommandInterpreter,
                         SourceInitFileInHomeDirectory,
                         (lldb::SBCommandReturnObject &));
    LLDB_REGISTER_METHOD(void, SBCommandInterpreter,
                         SourceInitFileInCurrentWorkingDirectory,
                         (lldb::SBCommandReturnObject &));
    LLDB_REGISTER_METHOD(lldb::SBBroadcaster, SBCommandInterpreter,
                         GetBroadcaster, ());
    LLDB_REGISTER_STATIC_METHOD(const char *, SBCommandInterpreter,
                                GetBroadcasterClass, ());
    LLDB_REGISTER_STATIC_METHOD(const char *, SBCommandInterpreter,
                                GetArgumentTypeAsCString,
                                (const lldb::CommandArgumentType));
    LLDB_REGISTER_STATIC_METHOD(const char *, SBCommandInterpreter,
                                GetArgumentDescriptionAsCString,
                                (const lldb::CommandArgumentType));
    LLDB_REGISTER_STATIC_METHOD(bool, SBCommandInterpreter,
                                EventIsCommandInterpreterEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_METHOD(lldb::SBCommand, SBCommandInterpreter,
                         AddMultiwordCommand, (const char *, const char *));
    LLDB_REGISTER_METHOD(
        lldb::SBCommand, SBCommandInterpreter, AddCommand,
        (const char *, lldb::SBCommandPluginInterface *, const char *));
    LLDB_REGISTER_METHOD(lldb::SBCommand, SBCommandInterpreter, AddCommand,
                         (const char *, lldb::SBCommandPluginInterface *,
                          const char *, const char *));
    LLDB_REGISTER_CONSTRUCTOR(SBCommand, ());
    LLDB_REGISTER_METHOD(bool, SBCommand, IsValid, ());
    LLDB_REGISTER_METHOD(const char *, SBCommand, GetName, ());
    LLDB_REGISTER_METHOD(const char *, SBCommand, GetHelp, ());
    LLDB_REGISTER_METHOD(const char *, SBCommand, GetHelpLong, ());
    LLDB_REGISTER_METHOD(void, SBCommand, SetHelp, (const char *));
    LLDB_REGISTER_METHOD(void, SBCommand, SetHelpLong, (const char *));
    LLDB_REGISTER_METHOD(lldb::SBCommand, SBCommand, AddMultiwordCommand,
                         (const char *, const char *));
    LLDB_REGISTER_METHOD(
        lldb::SBCommand, SBCommand, AddCommand,
        (const char *, lldb::SBCommandPluginInterface *, const char *));
    LLDB_REGISTER_METHOD(lldb::SBCommand, SBCommand, AddCommand,
                         (const char *, lldb::SBCommandPluginInterface *,
                          const char *, const char *));
    LLDB_REGISTER_METHOD(uint32_t, SBCommand, GetFlags, ());
    LLDB_REGISTER_METHOD(void, SBCommand, SetFlags, (uint32_t));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBCommandReturnObject, ());
    LLDB_REGISTER_CONSTRUCTOR(SBCommandReturnObject,
                              (const lldb::SBCommandReturnObject &));
    LLDB_REGISTER_CONSTRUCTOR(SBCommandReturnObject,
                              (lldb_private::CommandReturnObject *));
    LLDB_REGISTER_METHOD(lldb_private::CommandReturnObject *,
                         SBCommandReturnObject, Release, ());
    LLDB_REGISTER_METHOD(
        const lldb::SBCommandReturnObject &,
        SBCommandReturnObject, operator=,(const lldb::SBCommandReturnObject &));
    LLDB_REGISTER_METHOD_CONST(bool, SBCommandReturnObject, IsValid, ());
    LLDB_REGISTER_METHOD(const char *, SBCommandReturnObject, GetOutput, ());
    LLDB_REGISTER_METHOD(const char *, SBCommandReturnObject, GetError, ());
    LLDB_REGISTER_METHOD(size_t, SBCommandReturnObject, GetOutputSize, ());
    LLDB_REGISTER_METHOD(size_t, SBCommandReturnObject, GetErrorSize, ());
    LLDB_REGISTER_METHOD(size_t, SBCommandReturnObject, PutOutput, (FILE *));
    LLDB_REGISTER_METHOD(size_t, SBCommandReturnObject, PutError, (FILE *));
    LLDB_REGISTER_METHOD(void, SBCommandReturnObject, Clear, ());
    LLDB_REGISTER_METHOD(lldb::ReturnStatus, SBCommandReturnObject, GetStatus,
                         ());
    LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetStatus,
                         (lldb::ReturnStatus));
    LLDB_REGISTER_METHOD(bool, SBCommandReturnObject, Succeeded, ());
    LLDB_REGISTER_METHOD(bool, SBCommandReturnObject, HasResult, ());
    LLDB_REGISTER_METHOD(void, SBCommandReturnObject, AppendMessage,
                         (const char *));
    LLDB_REGISTER_METHOD(void, SBCommandReturnObject, AppendWarning,
                         (const char *));
    LLDB_REGISTER_METHOD(bool, SBCommandReturnObject, GetDescription,
                         (lldb::SBStream &));
    LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetImmediateOutputFile,
                         (FILE *));
    LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetImmediateErrorFile,
                         (FILE *));
    LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetImmediateOutputFile,
                         (FILE *, bool));
    LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetImmediateErrorFile,
                         (FILE *, bool));
    LLDB_REGISTER_METHOD(void, SBCommandReturnObject, PutCString,
                         (const char *, int));
    LLDB_REGISTER_METHOD(const char *, SBCommandReturnObject, GetOutput,
                         (bool));
    LLDB_REGISTER_METHOD(const char *, SBCommandReturnObject, GetError, (bool));
    LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetError,
                         (lldb::SBError &, const char *));
    LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetError, (const char *));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBCommunication, ());
    LLDB_REGISTER_CONSTRUCTOR(SBCommunication, (const char *));
    LLDB_REGISTER_METHOD_CONST(bool, SBCommunication, IsValid, ());
    LLDB_REGISTER_METHOD(bool, SBCommunication, GetCloseOnEOF, ());
    LLDB_REGISTER_METHOD(void, SBCommunication, SetCloseOnEOF, (bool));
    LLDB_REGISTER_METHOD(lldb::ConnectionStatus, SBCommunication, Connect,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::ConnectionStatus, SBCommunication,
                         AdoptFileDesriptor, (int, bool));
    LLDB_REGISTER_METHOD(lldb::ConnectionStatus, SBCommunication, Disconnect,
                         ());
    LLDB_REGISTER_METHOD_CONST(bool, SBCommunication, IsConnected, ());
    LLDB_REGISTER_METHOD(bool, SBCommunication, ReadThreadStart, ());
    LLDB_REGISTER_METHOD(bool, SBCommunication, ReadThreadStop, ());
    LLDB_REGISTER_METHOD(bool, SBCommunication, ReadThreadIsRunning, ());
    LLDB_REGISTER_METHOD(lldb::SBBroadcaster, SBCommunication, GetBroadcaster,
                         ());
    LLDB_REGISTER_STATIC_METHOD(const char *, SBCommunication,
                                GetBroadcasterClass, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBCompileUnit, ());
    LLDB_REGISTER_CONSTRUCTOR(SBCompileUnit, (const lldb::SBCompileUnit &));
    LLDB_REGISTER_METHOD(
        const lldb::SBCompileUnit &,
        SBCompileUnit, operator=,(const lldb::SBCompileUnit &));
    LLDB_REGISTER_METHOD_CONST(lldb::SBFileSpec, SBCompileUnit, GetFileSpec,
                               ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBCompileUnit, GetNumLineEntries, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBLineEntry, SBCompileUnit,
                               GetLineEntryAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBCompileUnit, FindLineEntryIndex,
                               (uint32_t, uint32_t, lldb::SBFileSpec *));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBCompileUnit, FindLineEntryIndex,
                               (uint32_t, uint32_t, lldb::SBFileSpec *, bool));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBCompileUnit, GetNumSupportFiles, ());
    LLDB_REGISTER_METHOD(lldb::SBTypeList, SBCompileUnit, GetTypes, (uint32_t));
    LLDB_REGISTER_METHOD_CONST(lldb::SBFileSpec, SBCompileUnit,
                               GetSupportFileAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(uint32_t, SBCompileUnit, FindSupportFileIndex,
                         (uint32_t, const lldb::SBFileSpec &, bool));
    LLDB_REGISTER_METHOD(lldb::LanguageType, SBCompileUnit, GetLanguage, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBCompileUnit, IsValid, ());
    LLDB_REGISTER_METHOD_CONST(
        bool, SBCompileUnit, operator==,(const lldb::SBCompileUnit &));
    LLDB_REGISTER_METHOD_CONST(
        bool, SBCompileUnit, operator!=,(const lldb::SBCompileUnit &));
    LLDB_REGISTER_METHOD(bool, SBCompileUnit, GetDescription,
                         (lldb::SBStream &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBData, ());
    LLDB_REGISTER_CONSTRUCTOR(SBData, (const lldb::SBData &));
    LLDB_REGISTER_METHOD(const lldb::SBData &,
                         SBData, operator=,(const lldb::SBData &));
    LLDB_REGISTER_METHOD(bool, SBData, IsValid, ());
    LLDB_REGISTER_METHOD(uint8_t, SBData, GetAddressByteSize, ());
    LLDB_REGISTER_METHOD(void, SBData, SetAddressByteSize, (uint8_t));
    LLDB_REGISTER_METHOD(void, SBData, Clear, ());
    LLDB_REGISTER_METHOD(size_t, SBData, GetByteSize, ());
    LLDB_REGISTER_METHOD(lldb::ByteOrder, SBData, GetByteOrder, ());
    LLDB_REGISTER_METHOD(void, SBData, SetByteOrder, (lldb::ByteOrder));
    LLDB_REGISTER_METHOD(float, SBData, GetFloat,
                         (lldb::SBError &, lldb::offset_t));
    LLDB_REGISTER_METHOD(double, SBData, GetDouble,
                         (lldb::SBError &, lldb::offset_t));
    LLDB_REGISTER_METHOD(long double, SBData, GetLongDouble,
                         (lldb::SBError &, lldb::offset_t));
    LLDB_REGISTER_METHOD(lldb::addr_t, SBData, GetAddress,
                         (lldb::SBError &, lldb::offset_t));
    LLDB_REGISTER_METHOD(uint8_t, SBData, GetUnsignedInt8,
                         (lldb::SBError &, lldb::offset_t));
    LLDB_REGISTER_METHOD(uint16_t, SBData, GetUnsignedInt16,
                         (lldb::SBError &, lldb::offset_t));
    LLDB_REGISTER_METHOD(uint32_t, SBData, GetUnsignedInt32,
                         (lldb::SBError &, lldb::offset_t));
    LLDB_REGISTER_METHOD(uint64_t, SBData, GetUnsignedInt64,
                         (lldb::SBError &, lldb::offset_t));
    LLDB_REGISTER_METHOD(int8_t, SBData, GetSignedInt8,
                         (lldb::SBError &, lldb::offset_t));
    LLDB_REGISTER_METHOD(int16_t, SBData, GetSignedInt16,
                         (lldb::SBError &, lldb::offset_t));
    LLDB_REGISTER_METHOD(int32_t, SBData, GetSignedInt32,
                         (lldb::SBError &, lldb::offset_t));
    LLDB_REGISTER_METHOD(int64_t, SBData, GetSignedInt64,
                         (lldb::SBError &, lldb::offset_t));
    LLDB_REGISTER_METHOD(const char *, SBData, GetString,
                         (lldb::SBError &, lldb::offset_t));
    LLDB_REGISTER_METHOD(bool, SBData, GetDescription,
                         (lldb::SBStream &, lldb::addr_t));
    LLDB_REGISTER_METHOD(bool, SBData, Append, (const lldb::SBData &));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBData, SBData, CreateDataFromCString,
                                (lldb::ByteOrder, uint32_t, const char *));
    LLDB_REGISTER_STATIC_METHOD(
        lldb::SBData, SBData, CreateDataFromUInt64Array,
        (lldb::ByteOrder, uint32_t, uint64_t *, size_t));
    LLDB_REGISTER_STATIC_METHOD(
        lldb::SBData, SBData, CreateDataFromUInt32Array,
        (lldb::ByteOrder, uint32_t, uint32_t *, size_t));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBData, SBData, CreateDataFromSInt64Array,
                                (lldb::ByteOrder, uint32_t, int64_t *, size_t));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBData, SBData, CreateDataFromSInt32Array,
                                (lldb::ByteOrder, uint32_t, int32_t *, size_t));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBData, SBData, CreateDataFromDoubleArray,
                                (lldb::ByteOrder, uint32_t, double *, size_t));
    LLDB_REGISTER_METHOD(bool, SBData, SetDataFromCString, (const char *));
    LLDB_REGISTER_METHOD(bool, SBData, SetDataFromUInt64Array,
                         (uint64_t *, size_t));
    LLDB_REGISTER_METHOD(bool, SBData, SetDataFromUInt32Array,
                         (uint32_t *, size_t));
    LLDB_REGISTER_METHOD(bool, SBData, SetDataFromSInt64Array,
                         (int64_t *, size_t));
    LLDB_REGISTER_METHOD(bool, SBData, SetDataFromSInt32Array,
                         (int32_t *, size_t));
    LLDB_REGISTER_METHOD(bool, SBData, SetDataFromDoubleArray,
                         (double *, size_t));
  }
  {
    LLDB_REGISTER_METHOD(void, SBInputReader, SetIsDone, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBInputReader, IsActive, ());
    LLDB_REGISTER_CONSTRUCTOR(SBDebugger, ());
    LLDB_REGISTER_CONSTRUCTOR(SBDebugger, (const lldb::DebuggerSP &));
    LLDB_REGISTER_CONSTRUCTOR(SBDebugger, (const lldb::SBDebugger &));
    LLDB_REGISTER_METHOD(lldb::SBDebugger &,
                         SBDebugger, operator=,(const lldb::SBDebugger &));
    LLDB_REGISTER_STATIC_METHOD(void, SBDebugger, Initialize, ());
    LLDB_REGISTER_STATIC_METHOD(lldb::SBError, SBDebugger,
                                InitializeWithErrorHandling, ());
    LLDB_REGISTER_STATIC_METHOD(void, SBDebugger, Terminate, ());
    LLDB_REGISTER_METHOD(void, SBDebugger, Clear, ());
    LLDB_REGISTER_STATIC_METHOD(lldb::SBDebugger, SBDebugger, Create, ());
    LLDB_REGISTER_STATIC_METHOD(lldb::SBDebugger, SBDebugger, Create, (bool));
    LLDB_REGISTER_STATIC_METHOD(void, SBDebugger, Destroy,
                                (lldb::SBDebugger &));
    LLDB_REGISTER_STATIC_METHOD(void, SBDebugger, MemoryPressureDetected, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBDebugger, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBDebugger, SetAsync, (bool));
    LLDB_REGISTER_METHOD(bool, SBDebugger, GetAsync, ());
    LLDB_REGISTER_METHOD(void, SBDebugger, SkipLLDBInitFiles, (bool));
    LLDB_REGISTER_METHOD(void, SBDebugger, SkipAppInitFiles, (bool));
    LLDB_REGISTER_METHOD(void, SBDebugger, SetInputFileHandle, (FILE *, bool));
    LLDB_REGISTER_METHOD(FILE *, SBDebugger, GetInputFileHandle, ());
    LLDB_REGISTER_METHOD(FILE *, SBDebugger, GetOutputFileHandle, ());
    LLDB_REGISTER_METHOD(FILE *, SBDebugger, GetErrorFileHandle, ());
    LLDB_REGISTER_METHOD(void, SBDebugger, SaveInputTerminalState, ());
    LLDB_REGISTER_METHOD(void, SBDebugger, RestoreInputTerminalState, ());
    LLDB_REGISTER_METHOD(lldb::SBCommandInterpreter, SBDebugger,
                         GetCommandInterpreter, ());
    LLDB_REGISTER_METHOD(void, SBDebugger, HandleCommand, (const char *));
    LLDB_REGISTER_METHOD(lldb::SBListener, SBDebugger, GetListener, ());
    LLDB_REGISTER_METHOD(
        void, SBDebugger, HandleProcessEvent,
        (const lldb::SBProcess &, const lldb::SBEvent &, FILE *, FILE *));
    LLDB_REGISTER_METHOD(lldb::SBSourceManager, SBDebugger, GetSourceManager,
                         ());
    LLDB_REGISTER_STATIC_METHOD(bool, SBDebugger, SetDefaultArchitecture,
                                (const char *));
    LLDB_REGISTER_METHOD(lldb::ScriptLanguage, SBDebugger, GetScriptingLanguage,
                         (const char *));
    LLDB_REGISTER_STATIC_METHOD(const char *, SBDebugger, GetVersionString, ());
    LLDB_REGISTER_STATIC_METHOD(const char *, SBDebugger, StateAsCString,
                                (lldb::StateType));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBStructuredData, SBDebugger,
                                GetBuildConfiguration, ());
    LLDB_REGISTER_STATIC_METHOD(bool, SBDebugger, StateIsRunningState,
                                (lldb::StateType));
    LLDB_REGISTER_STATIC_METHOD(bool, SBDebugger, StateIsStoppedState,
                                (lldb::StateType));
    LLDB_REGISTER_METHOD(
        lldb::SBTarget, SBDebugger, CreateTarget,
        (const char *, const char *, const char *, bool, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBTarget, SBDebugger,
                         CreateTargetWithFileAndTargetTriple,
                         (const char *, const char *));
    LLDB_REGISTER_METHOD(lldb::SBTarget, SBDebugger,
                         CreateTargetWithFileAndArch,
                         (const char *, const char *));
    LLDB_REGISTER_METHOD(lldb::SBTarget, SBDebugger, CreateTarget,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::SBTarget, SBDebugger, GetDummyTarget, ());
    LLDB_REGISTER_METHOD(bool, SBDebugger, DeleteTarget, (lldb::SBTarget &));
    LLDB_REGISTER_METHOD(lldb::SBTarget, SBDebugger, GetTargetAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(uint32_t, SBDebugger, GetIndexOfTarget,
                         (lldb::SBTarget));
    LLDB_REGISTER_METHOD(lldb::SBTarget, SBDebugger, FindTargetWithProcessID,
                         (lldb::pid_t));
    LLDB_REGISTER_METHOD(lldb::SBTarget, SBDebugger, FindTargetWithFileAndArch,
                         (const char *, const char *));
    LLDB_REGISTER_METHOD(uint32_t, SBDebugger, GetNumTargets, ());
    LLDB_REGISTER_METHOD(lldb::SBTarget, SBDebugger, GetSelectedTarget, ());
    LLDB_REGISTER_METHOD(void, SBDebugger, SetSelectedTarget,
                         (lldb::SBTarget &));
    LLDB_REGISTER_METHOD(lldb::SBPlatform, SBDebugger, GetSelectedPlatform, ());
    LLDB_REGISTER_METHOD(void, SBDebugger, SetSelectedPlatform,
                         (lldb::SBPlatform &));
    LLDB_REGISTER_METHOD(uint32_t, SBDebugger, GetNumPlatforms, ());
    LLDB_REGISTER_METHOD(lldb::SBPlatform, SBDebugger, GetPlatformAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(uint32_t, SBDebugger, GetNumAvailablePlatforms, ());
    LLDB_REGISTER_METHOD(lldb::SBStructuredData, SBDebugger,
                         GetAvailablePlatformInfoAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(void, SBDebugger, DispatchInputInterrupt, ());
    LLDB_REGISTER_METHOD(void, SBDebugger, DispatchInputEndOfFile, ());
    LLDB_REGISTER_METHOD(void, SBDebugger, PushInputReader,
                         (lldb::SBInputReader &));
    LLDB_REGISTER_METHOD(void, SBDebugger, RunCommandInterpreter, (bool, bool));
    LLDB_REGISTER_METHOD(void, SBDebugger, RunCommandInterpreter,
                         (bool, bool, lldb::SBCommandInterpreterRunOptions &,
                          int &, bool &, bool &));
    LLDB_REGISTER_METHOD(lldb::SBError, SBDebugger, RunREPL,
                         (lldb::LanguageType, const char *));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBDebugger, SBDebugger,
                                FindDebuggerWithID, (int));
    LLDB_REGISTER_METHOD(const char *, SBDebugger, GetInstanceName, ());
    LLDB_REGISTER_STATIC_METHOD(lldb::SBError, SBDebugger, SetInternalVariable,
                                (const char *, const char *, const char *));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBStringList, SBDebugger,
                                GetInternalVariableValue,
                                (const char *, const char *));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBDebugger, GetTerminalWidth, ());
    LLDB_REGISTER_METHOD(void, SBDebugger, SetTerminalWidth, (uint32_t));
    LLDB_REGISTER_METHOD_CONST(const char *, SBDebugger, GetPrompt, ());
    LLDB_REGISTER_METHOD(void, SBDebugger, SetPrompt, (const char *));
    LLDB_REGISTER_METHOD_CONST(const char *, SBDebugger, GetReproducerPath, ());
    LLDB_REGISTER_METHOD_CONST(lldb::ScriptLanguage, SBDebugger,
                               GetScriptLanguage, ());
    LLDB_REGISTER_METHOD(void, SBDebugger, SetScriptLanguage,
                         (lldb::ScriptLanguage));
    LLDB_REGISTER_METHOD(bool, SBDebugger, SetUseExternalEditor, (bool));
    LLDB_REGISTER_METHOD(bool, SBDebugger, GetUseExternalEditor, ());
    LLDB_REGISTER_METHOD(bool, SBDebugger, SetUseColor, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBDebugger, GetUseColor, ());
    LLDB_REGISTER_METHOD(bool, SBDebugger, GetDescription, (lldb::SBStream &));
    LLDB_REGISTER_METHOD(lldb::user_id_t, SBDebugger, GetID, ());
    LLDB_REGISTER_METHOD(lldb::SBError, SBDebugger, SetCurrentPlatform,
                         (const char *));
    LLDB_REGISTER_METHOD(bool, SBDebugger, SetCurrentPlatformSDKRoot,
                         (const char *));
    LLDB_REGISTER_METHOD_CONST(bool, SBDebugger, GetCloseInputOnEOF, ());
    LLDB_REGISTER_METHOD(void, SBDebugger, SetCloseInputOnEOF, (bool));
    LLDB_REGISTER_METHOD(lldb::SBTypeCategory, SBDebugger, GetCategory,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::SBTypeCategory, SBDebugger, GetCategory,
                         (lldb::LanguageType));
    LLDB_REGISTER_METHOD(lldb::SBTypeCategory, SBDebugger, CreateCategory,
                         (const char *));
    LLDB_REGISTER_METHOD(bool, SBDebugger, DeleteCategory, (const char *));
    LLDB_REGISTER_METHOD(uint32_t, SBDebugger, GetNumCategories, ());
    LLDB_REGISTER_METHOD(lldb::SBTypeCategory, SBDebugger, GetCategoryAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBTypeCategory, SBDebugger, GetDefaultCategory,
                         ());
    LLDB_REGISTER_METHOD(lldb::SBTypeFormat, SBDebugger, GetFormatForType,
                         (lldb::SBTypeNameSpecifier));
    LLDB_REGISTER_METHOD(lldb::SBTypeSummary, SBDebugger, GetSummaryForType,
                         (lldb::SBTypeNameSpecifier));
    LLDB_REGISTER_METHOD(lldb::SBTypeFilter, SBDebugger, GetFilterForType,
                         (lldb::SBTypeNameSpecifier));
    LLDB_REGISTER_METHOD(lldb::SBTypeSynthetic, SBDebugger, GetSyntheticForType,
                         (lldb::SBTypeNameSpecifier));
    LLDB_REGISTER_METHOD(bool, SBDebugger, EnableLog,
                         (const char *, const char **));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBDeclaration, ());
    LLDB_REGISTER_CONSTRUCTOR(SBDeclaration, (const lldb::SBDeclaration &));
    LLDB_REGISTER_METHOD(
        const lldb::SBDeclaration &,
        SBDeclaration, operator=,(const lldb::SBDeclaration &));
    LLDB_REGISTER_METHOD_CONST(bool, SBDeclaration, IsValid, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBFileSpec, SBDeclaration, GetFileSpec,
                               ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBDeclaration, GetLine, ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBDeclaration, GetColumn, ());
    LLDB_REGISTER_METHOD(void, SBDeclaration, SetFileSpec, (lldb::SBFileSpec));
    LLDB_REGISTER_METHOD(void, SBDeclaration, SetLine, (uint32_t));
    LLDB_REGISTER_METHOD(void, SBDeclaration, SetColumn, (uint32_t));
    LLDB_REGISTER_METHOD_CONST(
        bool, SBDeclaration, operator==,(const lldb::SBDeclaration &));
    LLDB_REGISTER_METHOD_CONST(
        bool, SBDeclaration, operator!=,(const lldb::SBDeclaration &));
    LLDB_REGISTER_METHOD(bool, SBDeclaration, GetDescription,
                         (lldb::SBStream &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBError, ());
    LLDB_REGISTER_CONSTRUCTOR(SBError, (const lldb::SBError &));
    LLDB_REGISTER_METHOD(const lldb::SBError &,
                         SBError, operator=,(const lldb::SBError &));
    LLDB_REGISTER_METHOD_CONST(const char *, SBError, GetCString, ());
    LLDB_REGISTER_METHOD(void, SBError, Clear, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBError, Fail, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBError, Success, ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBError, GetError, ());
    LLDB_REGISTER_METHOD_CONST(lldb::ErrorType, SBError, GetType, ());
    LLDB_REGISTER_METHOD(void, SBError, SetError, (uint32_t, lldb::ErrorType));
    LLDB_REGISTER_METHOD(void, SBError, SetErrorToErrno, ());
    LLDB_REGISTER_METHOD(void, SBError, SetErrorToGenericError, ());
    LLDB_REGISTER_METHOD(void, SBError, SetErrorString, (const char *));
    LLDB_REGISTER_METHOD_CONST(bool, SBError, IsValid, ());
    LLDB_REGISTER_METHOD(bool, SBError, GetDescription, (lldb::SBStream &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBEvent, ());
    LLDB_REGISTER_CONSTRUCTOR(SBEvent, (uint32_t, const char *, uint32_t));
    LLDB_REGISTER_CONSTRUCTOR(SBEvent, (lldb::EventSP &));
    LLDB_REGISTER_CONSTRUCTOR(SBEvent, (lldb_private::Event *));
    LLDB_REGISTER_CONSTRUCTOR(SBEvent, (const lldb::SBEvent &));
    LLDB_REGISTER_METHOD(const lldb::SBEvent &,
                         SBEvent, operator=,(const lldb::SBEvent &));
    LLDB_REGISTER_METHOD(const char *, SBEvent, GetDataFlavor, ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBEvent, GetType, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBBroadcaster, SBEvent, GetBroadcaster,
                               ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBEvent, GetBroadcasterClass, ());
    LLDB_REGISTER_METHOD(bool, SBEvent, BroadcasterMatchesPtr,
                         (const lldb::SBBroadcaster *));
    LLDB_REGISTER_METHOD(bool, SBEvent, BroadcasterMatchesRef,
                         (const lldb::SBBroadcaster &));
    LLDB_REGISTER_METHOD(void, SBEvent, Clear, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBEvent, IsValid, ());
    LLDB_REGISTER_STATIC_METHOD(const char *, SBEvent, GetCStringFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_METHOD(bool, SBEvent, GetDescription, (lldb::SBStream &));
    LLDB_REGISTER_METHOD_CONST(bool, SBEvent, GetDescription,
                               (lldb::SBStream &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBExecutionContext, ());
    LLDB_REGISTER_CONSTRUCTOR(SBExecutionContext,
                              (const lldb::SBExecutionContext &));
    LLDB_REGISTER_CONSTRUCTOR(SBExecutionContext,
                              (lldb::ExecutionContextRefSP));
    LLDB_REGISTER_CONSTRUCTOR(SBExecutionContext, (const lldb::SBTarget &));
    LLDB_REGISTER_CONSTRUCTOR(SBExecutionContext, (const lldb::SBProcess &));
    LLDB_REGISTER_CONSTRUCTOR(SBExecutionContext, (lldb::SBThread));
    LLDB_REGISTER_CONSTRUCTOR(SBExecutionContext, (const lldb::SBFrame &));
    LLDB_REGISTER_METHOD(
        const lldb::SBExecutionContext &,
        SBExecutionContext, operator=,(const lldb::SBExecutionContext &));
    LLDB_REGISTER_METHOD_CONST(lldb::SBTarget, SBExecutionContext, GetTarget,
                               ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBProcess, SBExecutionContext, GetProcess,
                               ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBThread, SBExecutionContext, GetThread,
                               ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBFrame, SBExecutionContext, GetFrame, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBExpressionOptions, ());
    LLDB_REGISTER_CONSTRUCTOR(SBExpressionOptions,
                              (const lldb::SBExpressionOptions &));
    LLDB_REGISTER_METHOD(
        const lldb::SBExpressionOptions &,
        SBExpressionOptions, operator=,(const lldb::SBExpressionOptions &));
    LLDB_REGISTER_METHOD_CONST(bool, SBExpressionOptions, GetCoerceResultToId,
                               ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetCoerceResultToId,
                         (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBExpressionOptions, GetUnwindOnError, ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetUnwindOnError, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBExpressionOptions, GetIgnoreBreakpoints,
                               ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetIgnoreBreakpoints,
                         (bool));
    LLDB_REGISTER_METHOD_CONST(lldb::DynamicValueType, SBExpressionOptions,
                               GetFetchDynamicValue, ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetFetchDynamicValue,
                         (lldb::DynamicValueType));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBExpressionOptions,
                               GetTimeoutInMicroSeconds, ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetTimeoutInMicroSeconds,
                         (uint32_t));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBExpressionOptions,
                               GetOneThreadTimeoutInMicroSeconds, ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions,
                         SetOneThreadTimeoutInMicroSeconds, (uint32_t));
    LLDB_REGISTER_METHOD_CONST(bool, SBExpressionOptions, GetTryAllThreads, ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetTryAllThreads, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBExpressionOptions, GetStopOthers, ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetStopOthers, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBExpressionOptions, GetTrapExceptions,
                               ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetTrapExceptions, (bool));
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetLanguage,
                         (lldb::LanguageType));
    LLDB_REGISTER_METHOD(bool, SBExpressionOptions, GetGenerateDebugInfo, ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetGenerateDebugInfo,
                         (bool));
    LLDB_REGISTER_METHOD(bool, SBExpressionOptions, GetSuppressPersistentResult,
                         ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetSuppressPersistentResult,
                         (bool));
    LLDB_REGISTER_METHOD_CONST(const char *, SBExpressionOptions, GetPrefix,
                               ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetPrefix, (const char *));
    LLDB_REGISTER_METHOD(bool, SBExpressionOptions, GetAutoApplyFixIts, ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetAutoApplyFixIts, (bool));
    LLDB_REGISTER_METHOD(bool, SBExpressionOptions, GetTopLevel, ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetTopLevel, (bool));
    LLDB_REGISTER_METHOD(bool, SBExpressionOptions, GetAllowJIT, ());
    LLDB_REGISTER_METHOD(void, SBExpressionOptions, SetAllowJIT, (bool));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBFileSpec, ());
    LLDB_REGISTER_CONSTRUCTOR(SBFileSpec, (const lldb::SBFileSpec &));
    LLDB_REGISTER_CONSTRUCTOR(SBFileSpec, (const char *));
    LLDB_REGISTER_CONSTRUCTOR(SBFileSpec, (const char *, bool));
    LLDB_REGISTER_METHOD(const lldb::SBFileSpec &,
                         SBFileSpec, operator=,(const lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD_CONST(bool, SBFileSpec, IsValid, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBFileSpec, Exists, ());
    LLDB_REGISTER_METHOD(bool, SBFileSpec, ResolveExecutableLocation, ());
    LLDB_REGISTER_STATIC_METHOD(int, SBFileSpec, ResolvePath,
                                (const char *, char *, size_t));
    LLDB_REGISTER_METHOD_CONST(const char *, SBFileSpec, GetFilename, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBFileSpec, GetDirectory, ());
    LLDB_REGISTER_METHOD(void, SBFileSpec, SetFilename, (const char *));
    LLDB_REGISTER_METHOD(void, SBFileSpec, SetDirectory, (const char *));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBFileSpec, GetPath, (char *, size_t));
    LLDB_REGISTER_METHOD_CONST(bool, SBFileSpec, GetDescription,
                               (lldb::SBStream &));
    LLDB_REGISTER_METHOD(void, SBFileSpec, AppendPathComponent, (const char *));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBFileSpecList, ());
    LLDB_REGISTER_CONSTRUCTOR(SBFileSpecList, (const lldb::SBFileSpecList &));
    LLDB_REGISTER_METHOD(
        const lldb::SBFileSpecList &,
        SBFileSpecList, operator=,(const lldb::SBFileSpecList &));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBFileSpecList, GetSize, ());
    LLDB_REGISTER_METHOD(void, SBFileSpecList, Append,
                         (const lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD(bool, SBFileSpecList, AppendIfUnique,
                         (const lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD(void, SBFileSpecList, Clear, ());
    LLDB_REGISTER_METHOD(uint32_t, SBFileSpecList, FindFileIndex,
                         (uint32_t, const lldb::SBFileSpec &, bool));
    LLDB_REGISTER_METHOD_CONST(const lldb::SBFileSpec, SBFileSpecList,
                               GetFileSpecAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD_CONST(bool, SBFileSpecList, GetDescription,
                               (lldb::SBStream &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBFrame, ());
    LLDB_REGISTER_CONSTRUCTOR(SBFrame, (const lldb::StackFrameSP &));
    LLDB_REGISTER_CONSTRUCTOR(SBFrame, (const lldb::SBFrame &));
    LLDB_REGISTER_METHOD(const lldb::SBFrame &,
                         SBFrame, operator=,(const lldb::SBFrame &));
    LLDB_REGISTER_METHOD_CONST(bool, SBFrame, IsValid, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBSymbolContext, SBFrame, GetSymbolContext,
                               (uint32_t));
    LLDB_REGISTER_METHOD_CONST(lldb::SBModule, SBFrame, GetModule, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBCompileUnit, SBFrame, GetCompileUnit,
                               ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBFunction, SBFrame, GetFunction, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBSymbol, SBFrame, GetSymbol, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBBlock, SBFrame, GetBlock, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBBlock, SBFrame, GetFrameBlock, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBLineEntry, SBFrame, GetLineEntry, ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBFrame, GetFrameID, ());
    LLDB_REGISTER_METHOD_CONST(lldb::addr_t, SBFrame, GetCFA, ());
    LLDB_REGISTER_METHOD_CONST(lldb::addr_t, SBFrame, GetPC, ());
    LLDB_REGISTER_METHOD(bool, SBFrame, SetPC, (lldb::addr_t));
    LLDB_REGISTER_METHOD_CONST(lldb::addr_t, SBFrame, GetSP, ());
    LLDB_REGISTER_METHOD_CONST(lldb::addr_t, SBFrame, GetFP, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBAddress, SBFrame, GetPCAddress, ());
    LLDB_REGISTER_METHOD(void, SBFrame, Clear, ());
    LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, GetValueForVariablePath,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, GetValueForVariablePath,
                         (const char *, lldb::DynamicValueType));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, FindVariable, (const char *));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, FindVariable,
                         (const char *, lldb::DynamicValueType));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, FindValue,
                         (const char *, lldb::ValueType));
    LLDB_REGISTER_METHOD(
        lldb::SBValue, SBFrame, FindValue,
        (const char *, lldb::ValueType, lldb::DynamicValueType));
    LLDB_REGISTER_METHOD_CONST(bool, SBFrame, IsEqual, (const lldb::SBFrame &));
    LLDB_REGISTER_METHOD_CONST(bool,
                               SBFrame, operator==,(const lldb::SBFrame &));
    LLDB_REGISTER_METHOD_CONST(bool,
                               SBFrame, operator!=,(const lldb::SBFrame &));
    LLDB_REGISTER_METHOD_CONST(lldb::SBThread, SBFrame, GetThread, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBFrame, Disassemble, ());
    LLDB_REGISTER_METHOD(lldb::SBValueList, SBFrame, GetVariables,
                         (bool, bool, bool, bool));
    LLDB_REGISTER_METHOD(lldb::SBValueList, SBFrame, GetVariables,
                         (bool, bool, bool, bool, lldb::DynamicValueType));
    LLDB_REGISTER_METHOD(lldb::SBValueList, SBFrame, GetVariables,
                         (const lldb::SBVariablesOptions &));
    LLDB_REGISTER_METHOD(lldb::SBValueList, SBFrame, GetRegisters, ());
    LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, FindRegister, (const char *));
    LLDB_REGISTER_METHOD(bool, SBFrame, GetDescription, (lldb::SBStream &));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, EvaluateExpression,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, EvaluateExpression,
                         (const char *, lldb::DynamicValueType));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, EvaluateExpression,
                         (const char *, lldb::DynamicValueType, bool));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, EvaluateExpression,
                         (const char *, const lldb::SBExpressionOptions &));
    LLDB_REGISTER_METHOD(bool, SBFrame, IsInlined, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBFrame, IsInlined, ());
    LLDB_REGISTER_METHOD(bool, SBFrame, IsArtificial, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBFrame, IsArtificial, ());
    LLDB_REGISTER_METHOD(const char *, SBFrame, GetFunctionName, ());
    LLDB_REGISTER_METHOD_CONST(lldb::LanguageType, SBFrame, GuessLanguage, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBFrame, GetFunctionName, ());
    LLDB_REGISTER_METHOD(const char *, SBFrame, GetDisplayFunctionName, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBFunction, ());
    LLDB_REGISTER_CONSTRUCTOR(SBFunction, (const lldb::SBFunction &));
    LLDB_REGISTER_METHOD(const lldb::SBFunction &,
                         SBFunction, operator=,(const lldb::SBFunction &));
    LLDB_REGISTER_METHOD_CONST(bool, SBFunction, IsValid, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBFunction, GetName, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBFunction, GetDisplayName, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBFunction, GetMangledName, ());
    LLDB_REGISTER_METHOD_CONST(
        bool, SBFunction, operator==,(const lldb::SBFunction &));
    LLDB_REGISTER_METHOD_CONST(
        bool, SBFunction, operator!=,(const lldb::SBFunction &));
    LLDB_REGISTER_METHOD(bool, SBFunction, GetDescription, (lldb::SBStream &));
    LLDB_REGISTER_METHOD(lldb::SBInstructionList, SBFunction, GetInstructions,
                         (lldb::SBTarget));
    LLDB_REGISTER_METHOD(lldb::SBInstructionList, SBFunction, GetInstructions,
                         (lldb::SBTarget, const char *));
    LLDB_REGISTER_METHOD(lldb::SBAddress, SBFunction, GetStartAddress, ());
    LLDB_REGISTER_METHOD(lldb::SBAddress, SBFunction, GetEndAddress, ());
    LLDB_REGISTER_METHOD(const char *, SBFunction, GetArgumentName, (uint32_t));
    LLDB_REGISTER_METHOD(uint32_t, SBFunction, GetPrologueByteSize, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBFunction, GetType, ());
    LLDB_REGISTER_METHOD(lldb::SBBlock, SBFunction, GetBlock, ());
    LLDB_REGISTER_METHOD(lldb::LanguageType, SBFunction, GetLanguage, ());
    LLDB_REGISTER_METHOD(bool, SBFunction, GetIsOptimized, ());
  }
  {
    LLDB_REGISTER_STATIC_METHOD(lldb::SBFileSpec, SBHostOS, GetProgramFileSpec,
                                ());
    LLDB_REGISTER_STATIC_METHOD(lldb::SBFileSpec, SBHostOS, GetLLDBPythonPath,
                                ());
    LLDB_REGISTER_STATIC_METHOD(lldb::SBFileSpec, SBHostOS, GetLLDBPath,
                                (lldb::PathType));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBFileSpec, SBHostOS,
                                GetUserHomeDirectory, ());
    LLDB_REGISTER_STATIC_METHOD(void, SBHostOS, ThreadCreated, (const char *));
    LLDB_REGISTER_STATIC_METHOD(bool, SBHostOS, ThreadCancel,
                                (lldb::thread_t, lldb::SBError *));
    LLDB_REGISTER_STATIC_METHOD(bool, SBHostOS, ThreadDetach,
                                (lldb::thread_t, lldb::SBError *));
    LLDB_REGISTER_STATIC_METHOD(
        bool, SBHostOS, ThreadJoin,
        (lldb::thread_t, lldb::thread_result_t *, lldb::SBError *));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBInstruction, ());
    LLDB_REGISTER_CONSTRUCTOR(SBInstruction, (const lldb::SBInstruction &));
    LLDB_REGISTER_METHOD(
        const lldb::SBInstruction &,
        SBInstruction, operator=,(const lldb::SBInstruction &));
    LLDB_REGISTER_METHOD(bool, SBInstruction, IsValid, ());
    LLDB_REGISTER_METHOD(lldb::SBAddress, SBInstruction, GetAddress, ());
    LLDB_REGISTER_METHOD(const char *, SBInstruction, GetMnemonic,
                         (lldb::SBTarget));
    LLDB_REGISTER_METHOD(const char *, SBInstruction, GetOperands,
                         (lldb::SBTarget));
    LLDB_REGISTER_METHOD(const char *, SBInstruction, GetComment,
                         (lldb::SBTarget));
    LLDB_REGISTER_METHOD(size_t, SBInstruction, GetByteSize, ());
    LLDB_REGISTER_METHOD(lldb::SBData, SBInstruction, GetData,
                         (lldb::SBTarget));
    LLDB_REGISTER_METHOD(bool, SBInstruction, DoesBranch, ());
    LLDB_REGISTER_METHOD(bool, SBInstruction, HasDelaySlot, ());
    LLDB_REGISTER_METHOD(bool, SBInstruction, CanSetBreakpoint, ());
    LLDB_REGISTER_METHOD(bool, SBInstruction, GetDescription,
                         (lldb::SBStream &));
    LLDB_REGISTER_METHOD(void, SBInstruction, Print, (FILE *));
    LLDB_REGISTER_METHOD(bool, SBInstruction, EmulateWithFrame,
                         (lldb::SBFrame &, uint32_t));
    LLDB_REGISTER_METHOD(bool, SBInstruction, DumpEmulation, (const char *));
    LLDB_REGISTER_METHOD(bool, SBInstruction, TestEmulation,
                         (lldb::SBStream &, const char *));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBInstructionList, ());
    LLDB_REGISTER_CONSTRUCTOR(SBInstructionList,
                              (const lldb::SBInstructionList &));
    LLDB_REGISTER_METHOD(
        const lldb::SBInstructionList &,
        SBInstructionList, operator=,(const lldb::SBInstructionList &));
    LLDB_REGISTER_METHOD_CONST(bool, SBInstructionList, IsValid, ());
    LLDB_REGISTER_METHOD(size_t, SBInstructionList, GetSize, ());
    LLDB_REGISTER_METHOD(lldb::SBInstruction, SBInstructionList,
                         GetInstructionAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(
        size_t, SBInstructionList, GetInstructionsCount,
        (const lldb::SBAddress &, const lldb::SBAddress &, bool));
    LLDB_REGISTER_METHOD(void, SBInstructionList, Clear, ());
    LLDB_REGISTER_METHOD(void, SBInstructionList, AppendInstruction,
                         (lldb::SBInstruction));
    LLDB_REGISTER_METHOD(void, SBInstructionList, Print, (FILE *));
    LLDB_REGISTER_METHOD(bool, SBInstructionList, GetDescription,
                         (lldb::SBStream &));
    LLDB_REGISTER_METHOD(bool, SBInstructionList,
                         DumpEmulationForAllInstructions, (const char *));
  }
  {
    LLDB_REGISTER_STATIC_METHOD(lldb::LanguageType, SBLanguageRuntime,
                                GetLanguageTypeFromString, (const char *));
    LLDB_REGISTER_STATIC_METHOD(const char *, SBLanguageRuntime,
                                GetNameForLanguageType, (lldb::LanguageType));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBLaunchInfo, (const char **));
    LLDB_REGISTER_METHOD(lldb::pid_t, SBLaunchInfo, GetProcessID, ());
    LLDB_REGISTER_METHOD(uint32_t, SBLaunchInfo, GetUserID, ());
    LLDB_REGISTER_METHOD(uint32_t, SBLaunchInfo, GetGroupID, ());
    LLDB_REGISTER_METHOD(bool, SBLaunchInfo, UserIDIsValid, ());
    LLDB_REGISTER_METHOD(bool, SBLaunchInfo, GroupIDIsValid, ());
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, SetUserID, (uint32_t));
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, SetGroupID, (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBFileSpec, SBLaunchInfo, GetExecutableFile, ());
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, SetExecutableFile,
                         (lldb::SBFileSpec, bool));
    LLDB_REGISTER_METHOD(lldb::SBListener, SBLaunchInfo, GetListener, ());
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, SetListener, (lldb::SBListener &));
    LLDB_REGISTER_METHOD(uint32_t, SBLaunchInfo, GetNumArguments, ());
    LLDB_REGISTER_METHOD(const char *, SBLaunchInfo, GetArgumentAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, SetArguments,
                         (const char **, bool));
    LLDB_REGISTER_METHOD(uint32_t, SBLaunchInfo, GetNumEnvironmentEntries, ());
    LLDB_REGISTER_METHOD(const char *, SBLaunchInfo, GetEnvironmentEntryAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, SetEnvironmentEntries,
                         (const char **, bool));
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, Clear, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBLaunchInfo, GetWorkingDirectory,
                               ());
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, SetWorkingDirectory,
                         (const char *));
    LLDB_REGISTER_METHOD(uint32_t, SBLaunchInfo, GetLaunchFlags, ());
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, SetLaunchFlags, (uint32_t));
    LLDB_REGISTER_METHOD(const char *, SBLaunchInfo, GetProcessPluginName, ());
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, SetProcessPluginName,
                         (const char *));
    LLDB_REGISTER_METHOD(const char *, SBLaunchInfo, GetShell, ());
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, SetShell, (const char *));
    LLDB_REGISTER_METHOD(bool, SBLaunchInfo, GetShellExpandArguments, ());
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, SetShellExpandArguments, (bool));
    LLDB_REGISTER_METHOD(uint32_t, SBLaunchInfo, GetResumeCount, ());
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, SetResumeCount, (uint32_t));
    LLDB_REGISTER_METHOD(bool, SBLaunchInfo, AddCloseFileAction, (int));
    LLDB_REGISTER_METHOD(bool, SBLaunchInfo, AddDuplicateFileAction,
                         (int, int));
    LLDB_REGISTER_METHOD(bool, SBLaunchInfo, AddOpenFileAction,
                         (int, const char *, bool, bool));
    LLDB_REGISTER_METHOD(bool, SBLaunchInfo, AddSuppressFileAction,
                         (int, bool, bool));
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, SetLaunchEventData,
                         (const char *));
    LLDB_REGISTER_METHOD_CONST(const char *, SBLaunchInfo, GetLaunchEventData,
                               ());
    LLDB_REGISTER_METHOD(void, SBLaunchInfo, SetDetachOnError, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBLaunchInfo, GetDetachOnError, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBLineEntry, ());
    LLDB_REGISTER_CONSTRUCTOR(SBLineEntry, (const lldb::SBLineEntry &));
    LLDB_REGISTER_METHOD(const lldb::SBLineEntry &,
                         SBLineEntry, operator=,(const lldb::SBLineEntry &));
    LLDB_REGISTER_METHOD_CONST(lldb::SBAddress, SBLineEntry, GetStartAddress,
                               ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBAddress, SBLineEntry, GetEndAddress, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBLineEntry, IsValid, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBFileSpec, SBLineEntry, GetFileSpec, ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBLineEntry, GetLine, ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBLineEntry, GetColumn, ());
    LLDB_REGISTER_METHOD(void, SBLineEntry, SetFileSpec, (lldb::SBFileSpec));
    LLDB_REGISTER_METHOD(void, SBLineEntry, SetLine, (uint32_t));
    LLDB_REGISTER_METHOD(void, SBLineEntry, SetColumn, (uint32_t));
    LLDB_REGISTER_METHOD_CONST(
        bool, SBLineEntry, operator==,(const lldb::SBLineEntry &));
    LLDB_REGISTER_METHOD_CONST(
        bool, SBLineEntry, operator!=,(const lldb::SBLineEntry &));
    LLDB_REGISTER_METHOD(bool, SBLineEntry, GetDescription, (lldb::SBStream &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBListener, ());
    LLDB_REGISTER_CONSTRUCTOR(SBListener, (const char *));
    LLDB_REGISTER_CONSTRUCTOR(SBListener, (const lldb::SBListener &));
    LLDB_REGISTER_METHOD(const lldb::SBListener &,
                         SBListener, operator=,(const lldb::SBListener &));
    LLDB_REGISTER_METHOD_CONST(bool, SBListener, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBListener, AddEvent, (const lldb::SBEvent &));
    LLDB_REGISTER_METHOD(void, SBListener, Clear, ());
    LLDB_REGISTER_METHOD(uint32_t, SBListener, StartListeningForEventClass,
                         (lldb::SBDebugger &, const char *, uint32_t));
    LLDB_REGISTER_METHOD(bool, SBListener, StopListeningForEventClass,
                         (lldb::SBDebugger &, const char *, uint32_t));
    LLDB_REGISTER_METHOD(uint32_t, SBListener, StartListeningForEvents,
                         (const lldb::SBBroadcaster &, uint32_t));
    LLDB_REGISTER_METHOD(bool, SBListener, StopListeningForEvents,
                         (const lldb::SBBroadcaster &, uint32_t));
    LLDB_REGISTER_METHOD(bool, SBListener, WaitForEvent,
                         (uint32_t, lldb::SBEvent &));
    LLDB_REGISTER_METHOD(
        bool, SBListener, WaitForEventForBroadcaster,
        (uint32_t, const lldb::SBBroadcaster &, lldb::SBEvent &));
    LLDB_REGISTER_METHOD(
        bool, SBListener, WaitForEventForBroadcasterWithType,
        (uint32_t, const lldb::SBBroadcaster &, uint32_t, lldb::SBEvent &));
    LLDB_REGISTER_METHOD(bool, SBListener, PeekAtNextEvent, (lldb::SBEvent &));
    LLDB_REGISTER_METHOD(bool, SBListener, PeekAtNextEventForBroadcaster,
                         (const lldb::SBBroadcaster &, lldb::SBEvent &));
    LLDB_REGISTER_METHOD(
        bool, SBListener, PeekAtNextEventForBroadcasterWithType,
        (const lldb::SBBroadcaster &, uint32_t, lldb::SBEvent &));
    LLDB_REGISTER_METHOD(bool, SBListener, GetNextEvent, (lldb::SBEvent &));
    LLDB_REGISTER_METHOD(bool, SBListener, GetNextEventForBroadcaster,
                         (const lldb::SBBroadcaster &, lldb::SBEvent &));
    LLDB_REGISTER_METHOD(
        bool, SBListener, GetNextEventForBroadcasterWithType,
        (const lldb::SBBroadcaster &, uint32_t, lldb::SBEvent &));
    LLDB_REGISTER_METHOD(bool, SBListener, HandleBroadcastEvent,
                         (const lldb::SBEvent &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBMemoryRegionInfo, ());
    LLDB_REGISTER_CONSTRUCTOR(SBMemoryRegionInfo,
                              (const lldb::SBMemoryRegionInfo &));
    LLDB_REGISTER_METHOD(
        const lldb::SBMemoryRegionInfo &,
        SBMemoryRegionInfo, operator=,(const lldb::SBMemoryRegionInfo &));
    LLDB_REGISTER_METHOD(void, SBMemoryRegionInfo, Clear, ());
    LLDB_REGISTER_METHOD_CONST(
        bool,
        SBMemoryRegionInfo, operator==,(const lldb::SBMemoryRegionInfo &));
    LLDB_REGISTER_METHOD_CONST(
        bool,
        SBMemoryRegionInfo, operator!=,(const lldb::SBMemoryRegionInfo &));
    LLDB_REGISTER_METHOD(lldb::addr_t, SBMemoryRegionInfo, GetRegionBase, ());
    LLDB_REGISTER_METHOD(lldb::addr_t, SBMemoryRegionInfo, GetRegionEnd, ());
    LLDB_REGISTER_METHOD(bool, SBMemoryRegionInfo, IsReadable, ());
    LLDB_REGISTER_METHOD(bool, SBMemoryRegionInfo, IsWritable, ());
    LLDB_REGISTER_METHOD(bool, SBMemoryRegionInfo, IsExecutable, ());
    LLDB_REGISTER_METHOD(bool, SBMemoryRegionInfo, IsMapped, ());
    LLDB_REGISTER_METHOD(const char *, SBMemoryRegionInfo, GetName, ());
    LLDB_REGISTER_METHOD(bool, SBMemoryRegionInfo, GetDescription,
                         (lldb::SBStream &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBMemoryRegionInfoList, ());
    LLDB_REGISTER_CONSTRUCTOR(SBMemoryRegionInfoList,
                              (const lldb::SBMemoryRegionInfoList &));
    LLDB_REGISTER_METHOD(
        const lldb::SBMemoryRegionInfoList &,
        SBMemoryRegionInfoList, operator=,(
                                    const lldb::SBMemoryRegionInfoList &));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBMemoryRegionInfoList, GetSize, ());
    LLDB_REGISTER_METHOD(bool, SBMemoryRegionInfoList, GetMemoryRegionAtIndex,
                         (uint32_t, lldb::SBMemoryRegionInfo &));
    LLDB_REGISTER_METHOD(void, SBMemoryRegionInfoList, Clear, ());
    LLDB_REGISTER_METHOD(void, SBMemoryRegionInfoList, Append,
                         (lldb::SBMemoryRegionInfo &));
    LLDB_REGISTER_METHOD(void, SBMemoryRegionInfoList, Append,
                         (lldb::SBMemoryRegionInfoList &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBModule, ());
    LLDB_REGISTER_CONSTRUCTOR(SBModule, (const lldb::SBModuleSpec &));
    LLDB_REGISTER_CONSTRUCTOR(SBModule, (const lldb::SBModule &));
    LLDB_REGISTER_CONSTRUCTOR(SBModule, (lldb::SBProcess &, lldb::addr_t));
    LLDB_REGISTER_METHOD(const lldb::SBModule &,
                         SBModule, operator=,(const lldb::SBModule &));
    LLDB_REGISTER_METHOD_CONST(bool, SBModule, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBModule, Clear, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBFileSpec, SBModule, GetFileSpec, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBFileSpec, SBModule, GetPlatformFileSpec,
                               ());
    LLDB_REGISTER_METHOD(bool, SBModule, SetPlatformFileSpec,
                         (const lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD(lldb::SBFileSpec, SBModule, GetRemoteInstallFileSpec,
                         ());
    LLDB_REGISTER_METHOD(bool, SBModule, SetRemoteInstallFileSpec,
                         (lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD_CONST(const char *, SBModule, GetUUIDString, ());
    LLDB_REGISTER_METHOD_CONST(bool,
                               SBModule, operator==,(const lldb::SBModule &));
    LLDB_REGISTER_METHOD_CONST(bool,
                               SBModule, operator!=,(const lldb::SBModule &));
    LLDB_REGISTER_METHOD(lldb::SBAddress, SBModule, ResolveFileAddress,
                         (lldb::addr_t));
    LLDB_REGISTER_METHOD(lldb::SBSymbolContext, SBModule,
                         ResolveSymbolContextForAddress,
                         (const lldb::SBAddress &, uint32_t));
    LLDB_REGISTER_METHOD(bool, SBModule, GetDescription, (lldb::SBStream &));
    LLDB_REGISTER_METHOD(uint32_t, SBModule, GetNumCompileUnits, ());
    LLDB_REGISTER_METHOD(lldb::SBCompileUnit, SBModule, GetCompileUnitAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBSymbolContextList, SBModule, FindCompileUnits,
                         (const lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD(size_t, SBModule, GetNumSymbols, ());
    LLDB_REGISTER_METHOD(lldb::SBSymbol, SBModule, GetSymbolAtIndex, (size_t));
    LLDB_REGISTER_METHOD(lldb::SBSymbol, SBModule, FindSymbol,
                         (const char *, lldb::SymbolType));
    LLDB_REGISTER_METHOD(lldb::SBSymbolContextList, SBModule, FindSymbols,
                         (const char *, lldb::SymbolType));
    LLDB_REGISTER_METHOD(size_t, SBModule, GetNumSections, ());
    LLDB_REGISTER_METHOD(lldb::SBSection, SBModule, GetSectionAtIndex,
                         (size_t));
    LLDB_REGISTER_METHOD(lldb::SBSymbolContextList, SBModule, FindFunctions,
                         (const char *, uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBValueList, SBModule, FindGlobalVariables,
                         (lldb::SBTarget &, const char *, uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBModule, FindFirstGlobalVariable,
                         (lldb::SBTarget &, const char *));
    LLDB_REGISTER_METHOD(lldb::SBType, SBModule, FindFirstType, (const char *));
    LLDB_REGISTER_METHOD(lldb::SBType, SBModule, GetBasicType,
                         (lldb::BasicType));
    LLDB_REGISTER_METHOD(lldb::SBTypeList, SBModule, FindTypes, (const char *));
    LLDB_REGISTER_METHOD(lldb::SBType, SBModule, GetTypeByID,
                         (lldb::user_id_t));
    LLDB_REGISTER_METHOD(lldb::SBTypeList, SBModule, GetTypes, (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBSection, SBModule, FindSection,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::ByteOrder, SBModule, GetByteOrder, ());
    LLDB_REGISTER_METHOD(const char *, SBModule, GetTriple, ());
    LLDB_REGISTER_METHOD(uint32_t, SBModule, GetAddressByteSize, ());
    LLDB_REGISTER_METHOD(uint32_t, SBModule, GetVersion,
                         (uint32_t *, uint32_t));
    LLDB_REGISTER_METHOD_CONST(lldb::SBFileSpec, SBModule, GetSymbolFileSpec,
                               ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBAddress, SBModule,
                               GetObjectFileHeaderAddress, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBAddress, SBModule,
                               GetObjectFileEntryPointAddress, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBModuleSpec, ());
    LLDB_REGISTER_CONSTRUCTOR(SBModuleSpec, (const lldb::SBModuleSpec &));
    LLDB_REGISTER_METHOD(const lldb::SBModuleSpec &,
                         SBModuleSpec, operator=,(const lldb::SBModuleSpec &));
    LLDB_REGISTER_METHOD_CONST(bool, SBModuleSpec, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBModuleSpec, Clear, ());
    LLDB_REGISTER_METHOD(lldb::SBFileSpec, SBModuleSpec, GetFileSpec, ());
    LLDB_REGISTER_METHOD(void, SBModuleSpec, SetFileSpec,
                         (const lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD(lldb::SBFileSpec, SBModuleSpec, GetPlatformFileSpec,
                         ());
    LLDB_REGISTER_METHOD(void, SBModuleSpec, SetPlatformFileSpec,
                         (const lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD(lldb::SBFileSpec, SBModuleSpec, GetSymbolFileSpec, ());
    LLDB_REGISTER_METHOD(void, SBModuleSpec, SetSymbolFileSpec,
                         (const lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD(const char *, SBModuleSpec, GetObjectName, ());
    LLDB_REGISTER_METHOD(void, SBModuleSpec, SetObjectName, (const char *));
    LLDB_REGISTER_METHOD(const char *, SBModuleSpec, GetTriple, ());
    LLDB_REGISTER_METHOD(void, SBModuleSpec, SetTriple, (const char *));
    LLDB_REGISTER_METHOD(size_t, SBModuleSpec, GetUUIDLength, ());
    LLDB_REGISTER_METHOD(bool, SBModuleSpec, GetDescription,
                         (lldb::SBStream &));
    LLDB_REGISTER_CONSTRUCTOR(SBModuleSpecList, ());
    LLDB_REGISTER_CONSTRUCTOR(SBModuleSpecList,
                              (const lldb::SBModuleSpecList &));
    LLDB_REGISTER_METHOD(
        lldb::SBModuleSpecList &,
        SBModuleSpecList, operator=,(const lldb::SBModuleSpecList &));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBModuleSpecList, SBModuleSpecList,
                                GetModuleSpecifications, (const char *));
    LLDB_REGISTER_METHOD(void, SBModuleSpecList, Append,
                         (const lldb::SBModuleSpec &));
    LLDB_REGISTER_METHOD(void, SBModuleSpecList, Append,
                         (const lldb::SBModuleSpecList &));
    LLDB_REGISTER_METHOD(size_t, SBModuleSpecList, GetSize, ());
    LLDB_REGISTER_METHOD(lldb::SBModuleSpec, SBModuleSpecList, GetSpecAtIndex,
                         (size_t));
    LLDB_REGISTER_METHOD(lldb::SBModuleSpec, SBModuleSpecList,
                         FindFirstMatchingSpec, (const lldb::SBModuleSpec &));
    LLDB_REGISTER_METHOD(lldb::SBModuleSpecList, SBModuleSpecList,
                         FindMatchingSpecs, (const lldb::SBModuleSpec &));
    LLDB_REGISTER_METHOD(bool, SBModuleSpecList, GetDescription,
                         (lldb::SBStream &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBPlatformConnectOptions, (const char *));
    LLDB_REGISTER_CONSTRUCTOR(SBPlatformConnectOptions,
                              (const lldb::SBPlatformConnectOptions &));
    LLDB_REGISTER_METHOD(
        void,
        SBPlatformConnectOptions, operator=,(
                                      const lldb::SBPlatformConnectOptions &));
    LLDB_REGISTER_METHOD(const char *, SBPlatformConnectOptions, GetURL, ());
    LLDB_REGISTER_METHOD(void, SBPlatformConnectOptions, SetURL,
                         (const char *));
    LLDB_REGISTER_METHOD(bool, SBPlatformConnectOptions, GetRsyncEnabled, ());
    LLDB_REGISTER_METHOD(void, SBPlatformConnectOptions, EnableRsync,
                         (const char *, const char *, bool));
    LLDB_REGISTER_METHOD(void, SBPlatformConnectOptions, DisableRsync, ());
    LLDB_REGISTER_METHOD(const char *, SBPlatformConnectOptions,
                         GetLocalCacheDirectory, ());
    LLDB_REGISTER_METHOD(void, SBPlatformConnectOptions, SetLocalCacheDirectory,
                         (const char *));
    LLDB_REGISTER_CONSTRUCTOR(SBPlatformShellCommand, (const char *));
    LLDB_REGISTER_CONSTRUCTOR(SBPlatformShellCommand,
                              (const lldb::SBPlatformShellCommand &));
    LLDB_REGISTER_METHOD(void, SBPlatformShellCommand, Clear, ());
    LLDB_REGISTER_METHOD(const char *, SBPlatformShellCommand, GetCommand, ());
    LLDB_REGISTER_METHOD(void, SBPlatformShellCommand, SetCommand,
                         (const char *));
    LLDB_REGISTER_METHOD(const char *, SBPlatformShellCommand,
                         GetWorkingDirectory, ());
    LLDB_REGISTER_METHOD(void, SBPlatformShellCommand, SetWorkingDirectory,
                         (const char *));
    LLDB_REGISTER_METHOD(uint32_t, SBPlatformShellCommand, GetTimeoutSeconds,
                         ());
    LLDB_REGISTER_METHOD(void, SBPlatformShellCommand, SetTimeoutSeconds,
                         (uint32_t));
    LLDB_REGISTER_METHOD(int, SBPlatformShellCommand, GetSignal, ());
    LLDB_REGISTER_METHOD(int, SBPlatformShellCommand, GetStatus, ());
    LLDB_REGISTER_METHOD(const char *, SBPlatformShellCommand, GetOutput, ());
    LLDB_REGISTER_CONSTRUCTOR(SBPlatform, ());
    LLDB_REGISTER_CONSTRUCTOR(SBPlatform, (const char *));
    LLDB_REGISTER_METHOD_CONST(bool, SBPlatform, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBPlatform, Clear, ());
    LLDB_REGISTER_METHOD(const char *, SBPlatform, GetName, ());
    LLDB_REGISTER_METHOD(const char *, SBPlatform, GetWorkingDirectory, ());
    LLDB_REGISTER_METHOD(bool, SBPlatform, SetWorkingDirectory, (const char *));
    LLDB_REGISTER_METHOD(lldb::SBError, SBPlatform, ConnectRemote,
                         (lldb::SBPlatformConnectOptions &));
    LLDB_REGISTER_METHOD(void, SBPlatform, DisconnectRemote, ());
    LLDB_REGISTER_METHOD(bool, SBPlatform, IsConnected, ());
    LLDB_REGISTER_METHOD(const char *, SBPlatform, GetTriple, ());
    LLDB_REGISTER_METHOD(const char *, SBPlatform, GetOSBuild, ());
    LLDB_REGISTER_METHOD(const char *, SBPlatform, GetOSDescription, ());
    LLDB_REGISTER_METHOD(const char *, SBPlatform, GetHostname, ());
    LLDB_REGISTER_METHOD(uint32_t, SBPlatform, GetOSMajorVersion, ());
    LLDB_REGISTER_METHOD(uint32_t, SBPlatform, GetOSMinorVersion, ());
    LLDB_REGISTER_METHOD(uint32_t, SBPlatform, GetOSUpdateVersion, ());
    LLDB_REGISTER_METHOD(lldb::SBError, SBPlatform, Get,
                         (lldb::SBFileSpec &, lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD(lldb::SBError, SBPlatform, Put,
                         (lldb::SBFileSpec &, lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD(lldb::SBError, SBPlatform, Install,
                         (lldb::SBFileSpec &, lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD(lldb::SBError, SBPlatform, Run,
                         (lldb::SBPlatformShellCommand &));
    LLDB_REGISTER_METHOD(lldb::SBError, SBPlatform, Launch,
                         (lldb::SBLaunchInfo &));
    LLDB_REGISTER_METHOD(lldb::SBError, SBPlatform, Kill, (const lldb::pid_t));
    LLDB_REGISTER_METHOD(lldb::SBError, SBPlatform, MakeDirectory,
                         (const char *, uint32_t));
    LLDB_REGISTER_METHOD(uint32_t, SBPlatform, GetFilePermissions,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::SBError, SBPlatform, SetFilePermissions,
                         (const char *, uint32_t));
    LLDB_REGISTER_METHOD_CONST(lldb::SBUnixSignals, SBPlatform, GetUnixSignals,
                               ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBProcess, ());
    LLDB_REGISTER_CONSTRUCTOR(SBProcess, (const lldb::SBProcess &));
    LLDB_REGISTER_CONSTRUCTOR(SBProcess, (const lldb::ProcessSP &));
    LLDB_REGISTER_METHOD(const lldb::SBProcess &,
                         SBProcess, operator=,(const lldb::SBProcess &));
    LLDB_REGISTER_STATIC_METHOD(const char *, SBProcess,
                                GetBroadcasterClassName, ());
    LLDB_REGISTER_METHOD(const char *, SBProcess, GetPluginName, ());
    LLDB_REGISTER_METHOD(const char *, SBProcess, GetShortPluginName, ());
    LLDB_REGISTER_METHOD(void, SBProcess, Clear, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBProcess, IsValid, ());
    LLDB_REGISTER_METHOD(bool, SBProcess, RemoteLaunch,
                         (const char **, const char **, const char *,
                          const char *, const char *, const char *, uint32_t,
                          bool, lldb::SBError &));
    LLDB_REGISTER_METHOD(bool, SBProcess, RemoteAttachToProcessWithID,
                         (lldb::pid_t, lldb::SBError &));
    LLDB_REGISTER_METHOD(uint32_t, SBProcess, GetNumThreads, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBThread, SBProcess, GetSelectedThread,
                               ());
    LLDB_REGISTER_METHOD(lldb::SBThread, SBProcess, CreateOSPluginThread,
                         (lldb::tid_t, lldb::addr_t));
    LLDB_REGISTER_METHOD_CONST(lldb::SBTarget, SBProcess, GetTarget, ());
    LLDB_REGISTER_METHOD(size_t, SBProcess, PutSTDIN, (const char *, size_t));
    LLDB_REGISTER_METHOD_CONST(size_t, SBProcess, GetSTDOUT, (char *, size_t));
    LLDB_REGISTER_METHOD_CONST(size_t, SBProcess, GetSTDERR, (char *, size_t));
    LLDB_REGISTER_METHOD_CONST(size_t, SBProcess, GetAsyncProfileData,
                               (char *, size_t));
    LLDB_REGISTER_METHOD(lldb::SBTrace, SBProcess, StartTrace,
                         (lldb::SBTraceOptions &, lldb::SBError &));
    LLDB_REGISTER_METHOD_CONST(void, SBProcess, ReportEventState,
                               (const lldb::SBEvent &, FILE *));
    LLDB_REGISTER_METHOD(
        void, SBProcess, AppendEventStateReport,
        (const lldb::SBEvent &, lldb::SBCommandReturnObject &));
    LLDB_REGISTER_METHOD(bool, SBProcess, SetSelectedThread,
                         (const lldb::SBThread &));
    LLDB_REGISTER_METHOD(bool, SBProcess, SetSelectedThreadByID, (lldb::tid_t));
    LLDB_REGISTER_METHOD(bool, SBProcess, SetSelectedThreadByIndexID,
                         (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBThread, SBProcess, GetThreadAtIndex, (size_t));
    LLDB_REGISTER_METHOD(uint32_t, SBProcess, GetNumQueues, ());
    LLDB_REGISTER_METHOD(lldb::SBQueue, SBProcess, GetQueueAtIndex, (size_t));
    LLDB_REGISTER_METHOD(uint32_t, SBProcess, GetStopID, (bool));
    LLDB_REGISTER_METHOD(lldb::SBEvent, SBProcess, GetStopEventForStopID,
                         (uint32_t));
    LLDB_REGISTER_METHOD(lldb::StateType, SBProcess, GetState, ());
    LLDB_REGISTER_METHOD(int, SBProcess, GetExitStatus, ());
    LLDB_REGISTER_METHOD(const char *, SBProcess, GetExitDescription, ());
    LLDB_REGISTER_METHOD(lldb::pid_t, SBProcess, GetProcessID, ());
    LLDB_REGISTER_METHOD(uint32_t, SBProcess, GetUniqueID, ());
    LLDB_REGISTER_METHOD_CONST(lldb::ByteOrder, SBProcess, GetByteOrder, ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBProcess, GetAddressByteSize, ());
    LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, Continue, ());
    LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, Destroy, ());
    LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, Stop, ());
    LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, Kill, ());
    LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, Detach, ());
    LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, Detach, (bool));
    LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, Signal, (int));
    LLDB_REGISTER_METHOD(lldb::SBUnixSignals, SBProcess, GetUnixSignals, ());
    LLDB_REGISTER_METHOD(void, SBProcess, SendAsyncInterrupt, ());
    LLDB_REGISTER_METHOD(lldb::SBThread, SBProcess, GetThreadByID,
                         (lldb::tid_t));
    LLDB_REGISTER_METHOD(lldb::SBThread, SBProcess, GetThreadByIndexID,
                         (uint32_t));
    LLDB_REGISTER_STATIC_METHOD(lldb::StateType, SBProcess, GetStateFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(bool, SBProcess, GetRestartedFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(size_t, SBProcess,
                                GetNumRestartedReasonsFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(const char *, SBProcess,
                                GetRestartedReasonAtIndexFromEvent,
                                (const lldb::SBEvent &, size_t));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBProcess, SBProcess, GetProcessFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(bool, SBProcess, GetInterruptedFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBStructuredData, SBProcess,
                                GetStructuredDataFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(bool, SBProcess, EventIsProcessEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(bool, SBProcess, EventIsStructuredDataEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_METHOD_CONST(lldb::SBBroadcaster, SBProcess, GetBroadcaster,
                               ());
    LLDB_REGISTER_STATIC_METHOD(const char *, SBProcess, GetBroadcasterClass,
                                ());
    LLDB_REGISTER_METHOD(uint64_t, SBProcess, ReadUnsignedFromMemory,
                         (lldb::addr_t, uint32_t, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::addr_t, SBProcess, ReadPointerFromMemory,
                         (lldb::addr_t, lldb::SBError &));
    LLDB_REGISTER_METHOD(bool, SBProcess, GetDescription, (lldb::SBStream &));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBProcess,
                               GetNumSupportedHardwareWatchpoints,
                               (lldb::SBError &));
    LLDB_REGISTER_METHOD(uint32_t, SBProcess, LoadImage,
                         (lldb::SBFileSpec &, lldb::SBError &));
    LLDB_REGISTER_METHOD(
        uint32_t, SBProcess, LoadImage,
        (const lldb::SBFileSpec &, const lldb::SBFileSpec &, lldb::SBError &));
    LLDB_REGISTER_METHOD(uint32_t, SBProcess, LoadImageUsingPaths,
                         (const lldb::SBFileSpec &, lldb::SBStringList &,
                          lldb::SBFileSpec &, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, UnloadImage, (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, SendEventData,
                         (const char *));
    LLDB_REGISTER_METHOD(uint32_t, SBProcess, GetNumExtendedBacktraceTypes, ());
    LLDB_REGISTER_METHOD(const char *, SBProcess,
                         GetExtendedBacktraceTypeAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBThreadCollection, SBProcess, GetHistoryThreads,
                         (lldb::addr_t));
    LLDB_REGISTER_METHOD(bool, SBProcess, IsInstrumentationRuntimePresent,
                         (lldb::InstrumentationRuntimeType));
    LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, SaveCore, (const char *));
    LLDB_REGISTER_METHOD(lldb::SBError, SBProcess, GetMemoryRegionInfo,
                         (lldb::addr_t, lldb::SBMemoryRegionInfo &));
    LLDB_REGISTER_METHOD(lldb::SBMemoryRegionInfoList, SBProcess,
                         GetMemoryRegions, ());
    LLDB_REGISTER_METHOD(lldb::SBProcessInfo, SBProcess, GetProcessInfo, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBProcessInfo, ());
    LLDB_REGISTER_CONSTRUCTOR(SBProcessInfo, (const lldb::SBProcessInfo &));
    LLDB_REGISTER_METHOD(
        lldb::SBProcessInfo &,
        SBProcessInfo, operator=,(const lldb::SBProcessInfo &));
    LLDB_REGISTER_METHOD_CONST(bool, SBProcessInfo, IsValid, ());
    LLDB_REGISTER_METHOD(const char *, SBProcessInfo, GetName, ());
    LLDB_REGISTER_METHOD(lldb::SBFileSpec, SBProcessInfo, GetExecutableFile,
                         ());
    LLDB_REGISTER_METHOD(lldb::pid_t, SBProcessInfo, GetProcessID, ());
    LLDB_REGISTER_METHOD(uint32_t, SBProcessInfo, GetUserID, ());
    LLDB_REGISTER_METHOD(uint32_t, SBProcessInfo, GetGroupID, ());
    LLDB_REGISTER_METHOD(bool, SBProcessInfo, UserIDIsValid, ());
    LLDB_REGISTER_METHOD(bool, SBProcessInfo, GroupIDIsValid, ());
    LLDB_REGISTER_METHOD(uint32_t, SBProcessInfo, GetEffectiveUserID, ());
    LLDB_REGISTER_METHOD(uint32_t, SBProcessInfo, GetEffectiveGroupID, ());
    LLDB_REGISTER_METHOD(bool, SBProcessInfo, EffectiveUserIDIsValid, ());
    LLDB_REGISTER_METHOD(bool, SBProcessInfo, EffectiveGroupIDIsValid, ());
    LLDB_REGISTER_METHOD(lldb::pid_t, SBProcessInfo, GetParentProcessID, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBQueue, ());
    LLDB_REGISTER_CONSTRUCTOR(SBQueue, (const lldb::QueueSP &));
    LLDB_REGISTER_CONSTRUCTOR(SBQueue, (const lldb::SBQueue &));
    LLDB_REGISTER_METHOD(const lldb::SBQueue &,
                         SBQueue, operator=,(const lldb::SBQueue &));
    LLDB_REGISTER_METHOD_CONST(bool, SBQueue, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBQueue, Clear, ());
    LLDB_REGISTER_METHOD_CONST(lldb::queue_id_t, SBQueue, GetQueueID, ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBQueue, GetIndexID, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBQueue, GetName, ());
    LLDB_REGISTER_METHOD(uint32_t, SBQueue, GetNumThreads, ());
    LLDB_REGISTER_METHOD(lldb::SBThread, SBQueue, GetThreadAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(uint32_t, SBQueue, GetNumPendingItems, ());
    LLDB_REGISTER_METHOD(lldb::SBQueueItem, SBQueue, GetPendingItemAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(uint32_t, SBQueue, GetNumRunningItems, ());
    LLDB_REGISTER_METHOD(lldb::SBProcess, SBQueue, GetProcess, ());
    LLDB_REGISTER_METHOD(lldb::QueueKind, SBQueue, GetKind, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBQueueItem, ());
    LLDB_REGISTER_CONSTRUCTOR(SBQueueItem, (const lldb::QueueItemSP &));
    LLDB_REGISTER_METHOD_CONST(bool, SBQueueItem, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBQueueItem, Clear, ());
    LLDB_REGISTER_METHOD(void, SBQueueItem, SetQueueItem,
                         (const lldb::QueueItemSP &));
    LLDB_REGISTER_METHOD_CONST(lldb::QueueItemKind, SBQueueItem, GetKind, ());
    LLDB_REGISTER_METHOD(void, SBQueueItem, SetKind, (lldb::QueueItemKind));
    LLDB_REGISTER_METHOD_CONST(lldb::SBAddress, SBQueueItem, GetAddress, ());
    LLDB_REGISTER_METHOD(void, SBQueueItem, SetAddress, (lldb::SBAddress));
    LLDB_REGISTER_METHOD(lldb::SBThread, SBQueueItem,
                         GetExtendedBacktraceThread, (const char *));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBSection, ());
    LLDB_REGISTER_CONSTRUCTOR(SBSection, (const lldb::SBSection &));
    LLDB_REGISTER_METHOD(const lldb::SBSection &,
                         SBSection, operator=,(const lldb::SBSection &));
    LLDB_REGISTER_METHOD_CONST(bool, SBSection, IsValid, ());
    LLDB_REGISTER_METHOD(const char *, SBSection, GetName, ());
    LLDB_REGISTER_METHOD(lldb::SBSection, SBSection, GetParent, ());
    LLDB_REGISTER_METHOD(lldb::SBSection, SBSection, FindSubSection,
                         (const char *));
    LLDB_REGISTER_METHOD(size_t, SBSection, GetNumSubSections, ());
    LLDB_REGISTER_METHOD(lldb::SBSection, SBSection, GetSubSectionAtIndex,
                         (size_t));
    LLDB_REGISTER_METHOD(lldb::addr_t, SBSection, GetFileAddress, ());
    LLDB_REGISTER_METHOD(lldb::addr_t, SBSection, GetLoadAddress,
                         (lldb::SBTarget &));
    LLDB_REGISTER_METHOD(lldb::addr_t, SBSection, GetByteSize, ());
    LLDB_REGISTER_METHOD(uint64_t, SBSection, GetFileOffset, ());
    LLDB_REGISTER_METHOD(uint64_t, SBSection, GetFileByteSize, ());
    LLDB_REGISTER_METHOD(lldb::SBData, SBSection, GetSectionData, ());
    LLDB_REGISTER_METHOD(lldb::SBData, SBSection, GetSectionData,
                         (uint64_t, uint64_t));
    LLDB_REGISTER_METHOD(lldb::SectionType, SBSection, GetSectionType, ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBSection, GetPermissions, ());
    LLDB_REGISTER_METHOD(uint32_t, SBSection, GetTargetByteSize, ());
    LLDB_REGISTER_METHOD(bool, SBSection, operator==,(const lldb::SBSection &));
    LLDB_REGISTER_METHOD(bool, SBSection, operator!=,(const lldb::SBSection &));
    LLDB_REGISTER_METHOD(bool, SBSection, GetDescription, (lldb::SBStream &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBSourceManager, (const lldb::SBDebugger &));
    LLDB_REGISTER_CONSTRUCTOR(SBSourceManager, (const lldb::SBTarget &));
    LLDB_REGISTER_CONSTRUCTOR(SBSourceManager, (const lldb::SBSourceManager &));
    LLDB_REGISTER_METHOD(
        const lldb::SBSourceManager &,
        SBSourceManager, operator=,(const lldb::SBSourceManager &));
    LLDB_REGISTER_METHOD(size_t, SBSourceManager,
                         DisplaySourceLinesWithLineNumbers,
                         (const lldb::SBFileSpec &, uint32_t, uint32_t,
                          uint32_t, const char *, lldb::SBStream &));
    LLDB_REGISTER_METHOD(size_t, SBSourceManager,
                         DisplaySourceLinesWithLineNumbersAndColumn,
                         (const lldb::SBFileSpec &, uint32_t, uint32_t,
                          uint32_t, uint32_t, const char *, lldb::SBStream &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBStream, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBStream, IsValid, ());
    LLDB_REGISTER_METHOD(const char *, SBStream, GetData, ());
    LLDB_REGISTER_METHOD(size_t, SBStream, GetSize, ());
    LLDB_REGISTER_METHOD(void, SBStream, RedirectToFile, (const char *, bool));
    LLDB_REGISTER_METHOD(void, SBStream, RedirectToFileHandle, (FILE *, bool));
    LLDB_REGISTER_METHOD(void, SBStream, RedirectToFileDescriptor, (int, bool));
    LLDB_REGISTER_METHOD(void, SBStream, Clear, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBStringList, ());
    LLDB_REGISTER_CONSTRUCTOR(SBStringList, (const lldb::SBStringList &));
    LLDB_REGISTER_METHOD(const lldb::SBStringList &,
                         SBStringList, operator=,(const lldb::SBStringList &));
    LLDB_REGISTER_METHOD_CONST(bool, SBStringList, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBStringList, AppendString, (const char *));
    LLDB_REGISTER_METHOD(void, SBStringList, AppendList, (const char **, int));
    LLDB_REGISTER_METHOD(void, SBStringList, AppendList,
                         (const lldb::SBStringList &));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBStringList, GetSize, ());
    LLDB_REGISTER_METHOD(const char *, SBStringList, GetStringAtIndex,
                         (size_t));
    LLDB_REGISTER_METHOD_CONST(const char *, SBStringList, GetStringAtIndex,
                               (size_t));
    LLDB_REGISTER_METHOD(void, SBStringList, Clear, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBStructuredData, ());
    LLDB_REGISTER_CONSTRUCTOR(SBStructuredData,
                              (const lldb::SBStructuredData &));
    LLDB_REGISTER_CONSTRUCTOR(SBStructuredData, (const lldb::EventSP &));
    LLDB_REGISTER_CONSTRUCTOR(SBStructuredData,
                              (lldb_private::StructuredDataImpl *));
    LLDB_REGISTER_METHOD(
        lldb::SBStructuredData &,
        SBStructuredData, operator=,(const lldb::SBStructuredData &));
    LLDB_REGISTER_METHOD(lldb::SBError, SBStructuredData, SetFromJSON,
                         (lldb::SBStream &));
    LLDB_REGISTER_METHOD_CONST(bool, SBStructuredData, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBStructuredData, Clear, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBError, SBStructuredData, GetAsJSON,
                               (lldb::SBStream &));
    LLDB_REGISTER_METHOD_CONST(lldb::SBError, SBStructuredData, GetDescription,
                               (lldb::SBStream &));
    LLDB_REGISTER_METHOD_CONST(lldb::StructuredDataType, SBStructuredData,
                               GetType, ());
    LLDB_REGISTER_METHOD_CONST(size_t, SBStructuredData, GetSize, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBStructuredData, GetKeys,
                               (lldb::SBStringList &));
    LLDB_REGISTER_METHOD_CONST(lldb::SBStructuredData, SBStructuredData,
                               GetValueForKey, (const char *));
    LLDB_REGISTER_METHOD_CONST(lldb::SBStructuredData, SBStructuredData,
                               GetItemAtIndex, (size_t));
    LLDB_REGISTER_METHOD_CONST(uint64_t, SBStructuredData, GetIntegerValue,
                               (uint64_t));
    LLDB_REGISTER_METHOD_CONST(double, SBStructuredData, GetFloatValue,
                               (double));
    LLDB_REGISTER_METHOD_CONST(bool, SBStructuredData, GetBooleanValue, (bool));
    LLDB_REGISTER_METHOD_CONST(size_t, SBStructuredData, GetStringValue,
                               (char *, size_t));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBSymbol, ());
    LLDB_REGISTER_CONSTRUCTOR(SBSymbol, (const lldb::SBSymbol &));
    LLDB_REGISTER_METHOD(const lldb::SBSymbol &,
                         SBSymbol, operator=,(const lldb::SBSymbol &));
    LLDB_REGISTER_METHOD_CONST(bool, SBSymbol, IsValid, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBSymbol, GetName, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBSymbol, GetDisplayName, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBSymbol, GetMangledName, ());
    LLDB_REGISTER_METHOD_CONST(bool,
                               SBSymbol, operator==,(const lldb::SBSymbol &));
    LLDB_REGISTER_METHOD_CONST(bool,
                               SBSymbol, operator!=,(const lldb::SBSymbol &));
    LLDB_REGISTER_METHOD(bool, SBSymbol, GetDescription, (lldb::SBStream &));
    LLDB_REGISTER_METHOD(lldb::SBInstructionList, SBSymbol, GetInstructions,
                         (lldb::SBTarget));
    LLDB_REGISTER_METHOD(lldb::SBInstructionList, SBSymbol, GetInstructions,
                         (lldb::SBTarget, const char *));
    LLDB_REGISTER_METHOD(lldb::SBAddress, SBSymbol, GetStartAddress, ());
    LLDB_REGISTER_METHOD(lldb::SBAddress, SBSymbol, GetEndAddress, ());
    LLDB_REGISTER_METHOD(uint32_t, SBSymbol, GetPrologueByteSize, ());
    LLDB_REGISTER_METHOD(lldb::SymbolType, SBSymbol, GetType, ());
    LLDB_REGISTER_METHOD(bool, SBSymbol, IsExternal, ());
    LLDB_REGISTER_METHOD(bool, SBSymbol, IsSynthetic, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBSymbolContext, ());
    LLDB_REGISTER_CONSTRUCTOR(SBSymbolContext,
                              (const lldb_private::SymbolContext *));
    LLDB_REGISTER_CONSTRUCTOR(SBSymbolContext, (const lldb::SBSymbolContext &));
    LLDB_REGISTER_METHOD(
        const lldb::SBSymbolContext &,
        SBSymbolContext, operator=,(const lldb::SBSymbolContext &));
    LLDB_REGISTER_METHOD_CONST(bool, SBSymbolContext, IsValid, ());
    LLDB_REGISTER_METHOD(lldb::SBModule, SBSymbolContext, GetModule, ());
    LLDB_REGISTER_METHOD(lldb::SBCompileUnit, SBSymbolContext, GetCompileUnit,
                         ());
    LLDB_REGISTER_METHOD(lldb::SBFunction, SBSymbolContext, GetFunction, ());
    LLDB_REGISTER_METHOD(lldb::SBBlock, SBSymbolContext, GetBlock, ());
    LLDB_REGISTER_METHOD(lldb::SBLineEntry, SBSymbolContext, GetLineEntry, ());
    LLDB_REGISTER_METHOD(lldb::SBSymbol, SBSymbolContext, GetSymbol, ());
    LLDB_REGISTER_METHOD(void, SBSymbolContext, SetModule, (lldb::SBModule));
    LLDB_REGISTER_METHOD(void, SBSymbolContext, SetCompileUnit,
                         (lldb::SBCompileUnit));
    LLDB_REGISTER_METHOD(void, SBSymbolContext, SetFunction,
                         (lldb::SBFunction));
    LLDB_REGISTER_METHOD(void, SBSymbolContext, SetBlock, (lldb::SBBlock));
    LLDB_REGISTER_METHOD(void, SBSymbolContext, SetLineEntry,
                         (lldb::SBLineEntry));
    LLDB_REGISTER_METHOD(void, SBSymbolContext, SetSymbol, (lldb::SBSymbol));
    LLDB_REGISTER_METHOD(bool, SBSymbolContext, GetDescription,
                         (lldb::SBStream &));
    LLDB_REGISTER_METHOD_CONST(lldb::SBSymbolContext, SBSymbolContext,
                               GetParentOfInlinedScope,
                               (const lldb::SBAddress &, lldb::SBAddress &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBSymbolContextList, ());
    LLDB_REGISTER_CONSTRUCTOR(SBSymbolContextList,
                              (const lldb::SBSymbolContextList &));
    LLDB_REGISTER_METHOD(
        const lldb::SBSymbolContextList &,
        SBSymbolContextList, operator=,(const lldb::SBSymbolContextList &));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBSymbolContextList, GetSize, ());
    LLDB_REGISTER_METHOD(lldb::SBSymbolContext, SBSymbolContextList,
                         GetContextAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(void, SBSymbolContextList, Clear, ());
    LLDB_REGISTER_METHOD(void, SBSymbolContextList, Append,
                         (lldb::SBSymbolContext &));
    LLDB_REGISTER_METHOD(void, SBSymbolContextList, Append,
                         (lldb::SBSymbolContextList &));
    LLDB_REGISTER_METHOD_CONST(bool, SBSymbolContextList, IsValid, ());
    LLDB_REGISTER_METHOD(bool, SBSymbolContextList, GetDescription,
                         (lldb::SBStream &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBTarget, ());
    LLDB_REGISTER_CONSTRUCTOR(SBTarget, (const lldb::SBTarget &));
    LLDB_REGISTER_CONSTRUCTOR(SBTarget, (const lldb::TargetSP &));
    LLDB_REGISTER_METHOD(const lldb::SBTarget &,
                         SBTarget, operator=,(const lldb::SBTarget &));
    LLDB_REGISTER_STATIC_METHOD(bool, SBTarget, EventIsTargetEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBTarget, SBTarget, GetTargetFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(uint32_t, SBTarget, GetNumModulesFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBModule, SBTarget,
                                GetModuleAtIndexFromEvent,
                                (const uint32_t, const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(const char *, SBTarget, GetBroadcasterClassName,
                                ());
    LLDB_REGISTER_METHOD_CONST(bool, SBTarget, IsValid, ());
    LLDB_REGISTER_METHOD(lldb::SBProcess, SBTarget, GetProcess, ());
    LLDB_REGISTER_METHOD(lldb::SBPlatform, SBTarget, GetPlatform, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBDebugger, SBTarget, GetDebugger, ());
    LLDB_REGISTER_METHOD(lldb::SBStructuredData, SBTarget, GetStatistics, ());
    LLDB_REGISTER_METHOD(void, SBTarget, SetCollectingStats, (bool));
    LLDB_REGISTER_METHOD(bool, SBTarget, GetCollectingStats, ());
    LLDB_REGISTER_METHOD(lldb::SBProcess, SBTarget, LoadCore, (const char *));
    LLDB_REGISTER_METHOD(lldb::SBProcess, SBTarget, LoadCore,
                         (const char *, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBProcess, SBTarget, LaunchSimple,
                         (const char **, const char **, const char *));
    LLDB_REGISTER_METHOD(lldb::SBError, SBTarget, Install, ());
    LLDB_REGISTER_METHOD(lldb::SBProcess, SBTarget, Launch,
                         (lldb::SBListener &, const char **, const char **,
                          const char *, const char *, const char *,
                          const char *, uint32_t, bool, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBProcess, SBTarget, Launch,
                         (lldb::SBLaunchInfo &, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBProcess, SBTarget, Attach,
                         (lldb::SBAttachInfo &, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBProcess, SBTarget, AttachToProcessWithID,
                         (lldb::SBListener &, lldb::pid_t, lldb::SBError &));
    LLDB_REGISTER_METHOD(
        lldb::SBProcess, SBTarget, AttachToProcessWithName,
        (lldb::SBListener &, const char *, bool, lldb::SBError &));
    LLDB_REGISTER_METHOD(
        lldb::SBProcess, SBTarget, ConnectRemote,
        (lldb::SBListener &, const char *, const char *, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBFileSpec, SBTarget, GetExecutable, ());
    LLDB_REGISTER_METHOD_CONST(bool,
                               SBTarget, operator==,(const lldb::SBTarget &));
    LLDB_REGISTER_METHOD_CONST(bool,
                               SBTarget, operator!=,(const lldb::SBTarget &));
    LLDB_REGISTER_METHOD(lldb::SBAddress, SBTarget, ResolveLoadAddress,
                         (lldb::addr_t));
    LLDB_REGISTER_METHOD(lldb::SBAddress, SBTarget, ResolveFileAddress,
                         (lldb::addr_t));
    LLDB_REGISTER_METHOD(lldb::SBAddress, SBTarget, ResolvePastLoadAddress,
                         (uint32_t, lldb::addr_t));
    LLDB_REGISTER_METHOD(lldb::SBSymbolContext, SBTarget,
                         ResolveSymbolContextForAddress,
                         (const lldb::SBAddress &, uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget,
                         BreakpointCreateByLocation, (const char *, uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget,
                         BreakpointCreateByLocation,
                         (const lldb::SBFileSpec &, uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget,
                         BreakpointCreateByLocation,
                         (const lldb::SBFileSpec &, uint32_t, lldb::addr_t));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget,
                         BreakpointCreateByLocation,
                         (const lldb::SBFileSpec &, uint32_t, lldb::addr_t,
                          lldb::SBFileSpecList &));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget,
                         BreakpointCreateByLocation,
                         (const lldb::SBFileSpec &, uint32_t, uint32_t,
                          lldb::addr_t, lldb::SBFileSpecList &));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget, BreakpointCreateByName,
                         (const char *, const char *));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget, BreakpointCreateByName,
                         (const char *, const lldb::SBFileSpecList &,
                          const lldb::SBFileSpecList &));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget, BreakpointCreateByName,
                         (const char *, uint32_t, const lldb::SBFileSpecList &,
                          const lldb::SBFileSpecList &));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget, BreakpointCreateByName,
                         (const char *, uint32_t, lldb::LanguageType,
                          const lldb::SBFileSpecList &,
                          const lldb::SBFileSpecList &));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget, BreakpointCreateByNames,
                         (const char **, uint32_t, uint32_t,
                          const lldb::SBFileSpecList &,
                          const lldb::SBFileSpecList &));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget, BreakpointCreateByNames,
                         (const char **, uint32_t, uint32_t, lldb::LanguageType,
                          const lldb::SBFileSpecList &,
                          const lldb::SBFileSpecList &));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget, BreakpointCreateByNames,
                         (const char **, uint32_t, uint32_t, lldb::LanguageType,
                          lldb::addr_t, const lldb::SBFileSpecList &,
                          const lldb::SBFileSpecList &));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget, BreakpointCreateByRegex,
                         (const char *, const char *));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget, BreakpointCreateByRegex,
                         (const char *, const lldb::SBFileSpecList &,
                          const lldb::SBFileSpecList &));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget, BreakpointCreateByRegex,
                         (const char *, lldb::LanguageType,
                          const lldb::SBFileSpecList &,
                          const lldb::SBFileSpecList &));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget,
                         BreakpointCreateByAddress, (lldb::addr_t));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget,
                         BreakpointCreateBySBAddress, (lldb::SBAddress &));
    LLDB_REGISTER_METHOD(
        lldb::SBBreakpoint, SBTarget, BreakpointCreateBySourceRegex,
        (const char *, const lldb::SBFileSpec &, const char *));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget,
                         BreakpointCreateBySourceRegex,
                         (const char *, const lldb::SBFileSpecList &,
                          const lldb::SBFileSpecList &));
    LLDB_REGISTER_METHOD(
        lldb::SBBreakpoint, SBTarget, BreakpointCreateBySourceRegex,
        (const char *, const lldb::SBFileSpecList &,
         const lldb::SBFileSpecList &, const lldb::SBStringList &));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget,
                         BreakpointCreateForException,
                         (lldb::LanguageType, bool, bool));
    LLDB_REGISTER_METHOD(
        lldb::SBBreakpoint, SBTarget, BreakpointCreateFromScript,
        (const char *, lldb::SBStructuredData &, const lldb::SBFileSpecList &,
         const lldb::SBFileSpecList &, bool));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBTarget, GetNumBreakpoints, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBBreakpoint, SBTarget,
                               GetBreakpointAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(bool, SBTarget, BreakpointDelete, (lldb::break_id_t));
    LLDB_REGISTER_METHOD(lldb::SBBreakpoint, SBTarget, FindBreakpointByID,
                         (lldb::break_id_t));
    LLDB_REGISTER_METHOD(bool, SBTarget, FindBreakpointsByName,
                         (const char *, lldb::SBBreakpointList &));
    LLDB_REGISTER_METHOD(void, SBTarget, GetBreakpointNames,
                         (lldb::SBStringList &));
    LLDB_REGISTER_METHOD(void, SBTarget, DeleteBreakpointName, (const char *));
    LLDB_REGISTER_METHOD(bool, SBTarget, EnableAllBreakpoints, ());
    LLDB_REGISTER_METHOD(bool, SBTarget, DisableAllBreakpoints, ());
    LLDB_REGISTER_METHOD(bool, SBTarget, DeleteAllBreakpoints, ());
    LLDB_REGISTER_METHOD(lldb::SBError, SBTarget, BreakpointsCreateFromFile,
                         (lldb::SBFileSpec &, lldb::SBBreakpointList &));
    LLDB_REGISTER_METHOD(
        lldb::SBError, SBTarget, BreakpointsCreateFromFile,
        (lldb::SBFileSpec &, lldb::SBStringList &, lldb::SBBreakpointList &));
    LLDB_REGISTER_METHOD(lldb::SBError, SBTarget, BreakpointsWriteToFile,
                         (lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD(lldb::SBError, SBTarget, BreakpointsWriteToFile,
                         (lldb::SBFileSpec &, lldb::SBBreakpointList &, bool));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBTarget, GetNumWatchpoints, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBWatchpoint, SBTarget,
                               GetWatchpointAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(bool, SBTarget, DeleteWatchpoint, (lldb::watch_id_t));
    LLDB_REGISTER_METHOD(lldb::SBWatchpoint, SBTarget, FindWatchpointByID,
                         (lldb::watch_id_t));
    LLDB_REGISTER_METHOD(lldb::SBWatchpoint, SBTarget, WatchAddress,
                         (lldb::addr_t, size_t, bool, bool, lldb::SBError &));
    LLDB_REGISTER_METHOD(bool, SBTarget, EnableAllWatchpoints, ());
    LLDB_REGISTER_METHOD(bool, SBTarget, DisableAllWatchpoints, ());
    LLDB_REGISTER_METHOD(lldb::SBValue, SBTarget, CreateValueFromAddress,
                         (const char *, lldb::SBAddress, lldb::SBType));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBTarget, CreateValueFromData,
                         (const char *, lldb::SBData, lldb::SBType));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBTarget, CreateValueFromExpression,
                         (const char *, const char *));
    LLDB_REGISTER_METHOD(bool, SBTarget, DeleteAllWatchpoints, ());
    LLDB_REGISTER_METHOD(void, SBTarget, AppendImageSearchPath,
                         (const char *, const char *, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBModule, SBTarget, AddModule,
                         (const char *, const char *, const char *));
    LLDB_REGISTER_METHOD(
        lldb::SBModule, SBTarget, AddModule,
        (const char *, const char *, const char *, const char *));
    LLDB_REGISTER_METHOD(lldb::SBModule, SBTarget, AddModule,
                         (const lldb::SBModuleSpec &));
    LLDB_REGISTER_METHOD(bool, SBTarget, AddModule, (lldb::SBModule &));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBTarget, GetNumModules, ());
    LLDB_REGISTER_METHOD(void, SBTarget, Clear, ());
    LLDB_REGISTER_METHOD(lldb::SBModule, SBTarget, FindModule,
                         (const lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD(lldb::SBSymbolContextList, SBTarget, FindCompileUnits,
                         (const lldb::SBFileSpec &));
    LLDB_REGISTER_METHOD(lldb::ByteOrder, SBTarget, GetByteOrder, ());
    LLDB_REGISTER_METHOD(const char *, SBTarget, GetTriple, ());
    LLDB_REGISTER_METHOD(uint32_t, SBTarget, GetDataByteSize, ());
    LLDB_REGISTER_METHOD(uint32_t, SBTarget, GetCodeByteSize, ());
    LLDB_REGISTER_METHOD(uint32_t, SBTarget, GetAddressByteSize, ());
    LLDB_REGISTER_METHOD(lldb::SBModule, SBTarget, GetModuleAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(bool, SBTarget, RemoveModule, (lldb::SBModule));
    LLDB_REGISTER_METHOD_CONST(lldb::SBBroadcaster, SBTarget, GetBroadcaster,
                               ());
    LLDB_REGISTER_METHOD(bool, SBTarget, GetDescription,
                         (lldb::SBStream &, lldb::DescriptionLevel));
    LLDB_REGISTER_METHOD(lldb::SBSymbolContextList, SBTarget, FindFunctions,
                         (const char *, uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBSymbolContextList, SBTarget,
                         FindGlobalFunctions,
                         (const char *, uint32_t, lldb::MatchType));
    LLDB_REGISTER_METHOD(lldb::SBType, SBTarget, FindFirstType, (const char *));
    LLDB_REGISTER_METHOD(lldb::SBType, SBTarget, GetBasicType,
                         (lldb::BasicType));
    LLDB_REGISTER_METHOD(lldb::SBTypeList, SBTarget, FindTypes, (const char *));
    LLDB_REGISTER_METHOD(lldb::SBValueList, SBTarget, FindGlobalVariables,
                         (const char *, uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBValueList, SBTarget, FindGlobalVariables,
                         (const char *, uint32_t, lldb::MatchType));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBTarget, FindFirstGlobalVariable,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::SBSourceManager, SBTarget, GetSourceManager, ());
    LLDB_REGISTER_METHOD(lldb::SBInstructionList, SBTarget, ReadInstructions,
                         (lldb::SBAddress, uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBInstructionList, SBTarget, ReadInstructions,
                         (lldb::SBAddress, uint32_t, const char *));
    LLDB_REGISTER_METHOD(lldb::SBError, SBTarget, SetSectionLoadAddress,
                         (lldb::SBSection, lldb::addr_t));
    LLDB_REGISTER_METHOD(lldb::SBError, SBTarget, ClearSectionLoadAddress,
                         (lldb::SBSection));
    LLDB_REGISTER_METHOD(lldb::SBError, SBTarget, SetModuleLoadAddress,
                         (lldb::SBModule, int64_t));
    LLDB_REGISTER_METHOD(lldb::SBError, SBTarget, ClearModuleLoadAddress,
                         (lldb::SBModule));
    LLDB_REGISTER_METHOD(lldb::SBSymbolContextList, SBTarget, FindSymbols,
                         (const char *, lldb::SymbolType));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBTarget, EvaluateExpression,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBTarget, EvaluateExpression,
                         (const char *, const lldb::SBExpressionOptions &));
    LLDB_REGISTER_METHOD(lldb::addr_t, SBTarget, GetStackRedZoneSize, ());
    LLDB_REGISTER_METHOD_CONST(lldb::SBLaunchInfo, SBTarget, GetLaunchInfo, ());
    LLDB_REGISTER_METHOD(void, SBTarget, SetLaunchInfo,
                         (const lldb::SBLaunchInfo &));
  }
  {
    LLDB_REGISTER_STATIC_METHOD(const char *, SBThread, GetBroadcasterClassName,
                                ());
    LLDB_REGISTER_CONSTRUCTOR(SBThread, ());
    LLDB_REGISTER_CONSTRUCTOR(SBThread, (const lldb::ThreadSP &));
    LLDB_REGISTER_CONSTRUCTOR(SBThread, (const lldb::SBThread &));
    LLDB_REGISTER_METHOD(const lldb::SBThread &,
                         SBThread, operator=,(const lldb::SBThread &));
    LLDB_REGISTER_METHOD_CONST(lldb::SBQueue, SBThread, GetQueue, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBThread, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBThread, Clear, ());
    LLDB_REGISTER_METHOD(lldb::StopReason, SBThread, GetStopReason, ());
    LLDB_REGISTER_METHOD(size_t, SBThread, GetStopReasonDataCount, ());
    LLDB_REGISTER_METHOD(uint64_t, SBThread, GetStopReasonDataAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(bool, SBThread, GetStopReasonExtendedInfoAsJSON,
                         (lldb::SBStream &));
    LLDB_REGISTER_METHOD(lldb::SBThreadCollection, SBThread,
                         GetStopReasonExtendedBacktraces,
                         (lldb::InstrumentationRuntimeType));
    LLDB_REGISTER_METHOD(size_t, SBThread, GetStopDescription,
                         (char *, size_t));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBThread, GetStopReturnValue, ());
    LLDB_REGISTER_METHOD_CONST(lldb::tid_t, SBThread, GetThreadID, ());
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBThread, GetIndexID, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBThread, GetName, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBThread, GetQueueName, ());
    LLDB_REGISTER_METHOD_CONST(lldb::queue_id_t, SBThread, GetQueueID, ());
    LLDB_REGISTER_METHOD(bool, SBThread, GetInfoItemByPathAsString,
                         (const char *, lldb::SBStream &));
    LLDB_REGISTER_METHOD(void, SBThread, StepOver, (lldb::RunMode));
    LLDB_REGISTER_METHOD(void, SBThread, StepOver,
                         (lldb::RunMode, lldb::SBError &));
    LLDB_REGISTER_METHOD(void, SBThread, StepInto, (lldb::RunMode));
    LLDB_REGISTER_METHOD(void, SBThread, StepInto,
                         (const char *, lldb::RunMode));
    LLDB_REGISTER_METHOD(
        void, SBThread, StepInto,
        (const char *, uint32_t, lldb::SBError &, lldb::RunMode));
    LLDB_REGISTER_METHOD(void, SBThread, StepOut, ());
    LLDB_REGISTER_METHOD(void, SBThread, StepOut, (lldb::SBError &));
    LLDB_REGISTER_METHOD(void, SBThread, StepOutOfFrame, (lldb::SBFrame &));
    LLDB_REGISTER_METHOD(void, SBThread, StepOutOfFrame,
                         (lldb::SBFrame &, lldb::SBError &));
    LLDB_REGISTER_METHOD(void, SBThread, StepInstruction, (bool));
    LLDB_REGISTER_METHOD(void, SBThread, StepInstruction,
                         (bool, lldb::SBError &));
    LLDB_REGISTER_METHOD(void, SBThread, RunToAddress, (lldb::addr_t));
    LLDB_REGISTER_METHOD(void, SBThread, RunToAddress,
                         (lldb::addr_t, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBError, SBThread, StepOverUntil,
                         (lldb::SBFrame &, lldb::SBFileSpec &, uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBError, SBThread, StepUsingScriptedThreadPlan,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::SBError, SBThread, StepUsingScriptedThreadPlan,
                         (const char *, bool));
    LLDB_REGISTER_METHOD(lldb::SBError, SBThread, JumpToLine,
                         (lldb::SBFileSpec &, uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBError, SBThread, ReturnFromFrame,
                         (lldb::SBFrame &, lldb::SBValue &));
    LLDB_REGISTER_METHOD(lldb::SBError, SBThread, UnwindInnermostExpression,
                         ());
    LLDB_REGISTER_METHOD(bool, SBThread, Suspend, ());
    LLDB_REGISTER_METHOD(bool, SBThread, Suspend, (lldb::SBError &));
    LLDB_REGISTER_METHOD(bool, SBThread, Resume, ());
    LLDB_REGISTER_METHOD(bool, SBThread, Resume, (lldb::SBError &));
    LLDB_REGISTER_METHOD(bool, SBThread, IsSuspended, ());
    LLDB_REGISTER_METHOD(bool, SBThread, IsStopped, ());
    LLDB_REGISTER_METHOD(lldb::SBProcess, SBThread, GetProcess, ());
    LLDB_REGISTER_METHOD(uint32_t, SBThread, GetNumFrames, ());
    LLDB_REGISTER_METHOD(lldb::SBFrame, SBThread, GetFrameAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBFrame, SBThread, GetSelectedFrame, ());
    LLDB_REGISTER_METHOD(lldb::SBFrame, SBThread, SetSelectedFrame, (uint32_t));
    LLDB_REGISTER_STATIC_METHOD(bool, SBThread, EventIsThreadEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBFrame, SBThread, GetStackFrameFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBThread, SBThread, GetThreadFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_METHOD_CONST(bool,
                               SBThread, operator==,(const lldb::SBThread &));
    LLDB_REGISTER_METHOD_CONST(bool,
                               SBThread, operator!=,(const lldb::SBThread &));
    LLDB_REGISTER_METHOD_CONST(bool, SBThread, GetStatus, (lldb::SBStream &));
    LLDB_REGISTER_METHOD_CONST(bool, SBThread, GetDescription,
                               (lldb::SBStream &));
    LLDB_REGISTER_METHOD_CONST(bool, SBThread, GetDescription,
                               (lldb::SBStream &, bool));
    LLDB_REGISTER_METHOD(lldb::SBThread, SBThread, GetExtendedBacktraceThread,
                         (const char *));
    LLDB_REGISTER_METHOD(uint32_t, SBThread,
                         GetExtendedBacktraceOriginatingIndexID, ());
    LLDB_REGISTER_METHOD(lldb::SBValue, SBThread, GetCurrentException, ());
    LLDB_REGISTER_METHOD(lldb::SBThread, SBThread, GetCurrentExceptionBacktrace,
                         ());
    LLDB_REGISTER_METHOD(bool, SBThread, SafeToCallFunctions, ());
    LLDB_REGISTER_METHOD(lldb_private::Thread *, SBThread, operator->,());
    LLDB_REGISTER_METHOD(lldb_private::Thread *, SBThread, get, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBThreadCollection, ());
    LLDB_REGISTER_CONSTRUCTOR(SBThreadCollection,
                              (const lldb::SBThreadCollection &));
    LLDB_REGISTER_METHOD(
        const lldb::SBThreadCollection &,
        SBThreadCollection, operator=,(const lldb::SBThreadCollection &));
    LLDB_REGISTER_METHOD_CONST(bool, SBThreadCollection, IsValid, ());
    LLDB_REGISTER_METHOD(size_t, SBThreadCollection, GetSize, ());
    LLDB_REGISTER_METHOD(lldb::SBThread, SBThreadCollection, GetThreadAtIndex,
                         (size_t));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBThreadPlan, ());
    LLDB_REGISTER_CONSTRUCTOR(SBThreadPlan, (const lldb::ThreadPlanSP &));
    LLDB_REGISTER_CONSTRUCTOR(SBThreadPlan, (const lldb::SBThreadPlan &));
    LLDB_REGISTER_CONSTRUCTOR(SBThreadPlan, (lldb::SBThread &, const char *));
    LLDB_REGISTER_METHOD(const lldb::SBThreadPlan &,
                         SBThreadPlan, operator=,(const lldb::SBThreadPlan &));
    LLDB_REGISTER_METHOD(lldb_private::ThreadPlan *, SBThreadPlan, get, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBThreadPlan, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBThreadPlan, Clear, ());
    LLDB_REGISTER_METHOD(lldb::StopReason, SBThreadPlan, GetStopReason, ());
    LLDB_REGISTER_METHOD(size_t, SBThreadPlan, GetStopReasonDataCount, ());
    LLDB_REGISTER_METHOD(uint64_t, SBThreadPlan, GetStopReasonDataAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD_CONST(lldb::SBThread, SBThreadPlan, GetThread, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBThreadPlan, GetDescription,
                               (lldb::SBStream &));
    LLDB_REGISTER_METHOD(void, SBThreadPlan, SetPlanComplete, (bool));
    LLDB_REGISTER_METHOD(bool, SBThreadPlan, IsPlanComplete, ());
    LLDB_REGISTER_METHOD(bool, SBThreadPlan, IsPlanStale, ());
    LLDB_REGISTER_METHOD(bool, SBThreadPlan, IsValid, ());
    LLDB_REGISTER_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                         QueueThreadPlanForStepOverRange,
                         (lldb::SBAddress &, lldb::addr_t));
    LLDB_REGISTER_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                         QueueThreadPlanForStepOverRange,
                         (lldb::SBAddress &, lldb::addr_t, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                         QueueThreadPlanForStepInRange,
                         (lldb::SBAddress &, lldb::addr_t));
    LLDB_REGISTER_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                         QueueThreadPlanForStepInRange,
                         (lldb::SBAddress &, lldb::addr_t, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                         QueueThreadPlanForStepOut, (uint32_t, bool));
    LLDB_REGISTER_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                         QueueThreadPlanForStepOut,
                         (uint32_t, bool, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                         QueueThreadPlanForRunToAddress, (lldb::SBAddress));
    LLDB_REGISTER_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                         QueueThreadPlanForRunToAddress,
                         (lldb::SBAddress, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                         QueueThreadPlanForStepScripted, (const char *));
    LLDB_REGISTER_METHOD(lldb::SBThreadPlan, SBThreadPlan,
                         QueueThreadPlanForStepScripted,
                         (const char *, lldb::SBError &));
  }
  {
    LLDB_REGISTER_METHOD(void, SBTrace, StopTrace,
                         (lldb::SBError &, lldb::tid_t));
    LLDB_REGISTER_METHOD(void, SBTrace, GetTraceConfig,
                         (lldb::SBTraceOptions &, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::user_id_t, SBTrace, GetTraceUID, ());
    LLDB_REGISTER_CONSTRUCTOR(SBTrace, ());
    LLDB_REGISTER_METHOD(bool, SBTrace, IsValid, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBTraceOptions, ());
    LLDB_REGISTER_METHOD_CONST(lldb::TraceType, SBTraceOptions, getType, ());
    LLDB_REGISTER_METHOD_CONST(uint64_t, SBTraceOptions, getTraceBufferSize,
                               ());
    LLDB_REGISTER_METHOD(lldb::SBStructuredData, SBTraceOptions, getTraceParams,
                         (lldb::SBError &));
    LLDB_REGISTER_METHOD_CONST(uint64_t, SBTraceOptions, getMetaDataBufferSize,
                               ());
    LLDB_REGISTER_METHOD(void, SBTraceOptions, setTraceParams,
                         (lldb::SBStructuredData &));
    LLDB_REGISTER_METHOD(void, SBTraceOptions, setType, (lldb::TraceType));
    LLDB_REGISTER_METHOD(void, SBTraceOptions, setTraceBufferSize, (uint64_t));
    LLDB_REGISTER_METHOD(void, SBTraceOptions, setMetaDataBufferSize,
                         (uint64_t));
    LLDB_REGISTER_METHOD(bool, SBTraceOptions, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBTraceOptions, setThreadID, (lldb::tid_t));
    LLDB_REGISTER_METHOD(lldb::tid_t, SBTraceOptions, getThreadID, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBType, ());
    LLDB_REGISTER_CONSTRUCTOR(SBType, (const lldb::SBType &));
    LLDB_REGISTER_METHOD(bool, SBType, operator==,(lldb::SBType &));
    LLDB_REGISTER_METHOD(bool, SBType, operator!=,(lldb::SBType &));
    LLDB_REGISTER_METHOD(lldb::SBType &,
                         SBType, operator=,(const lldb::SBType &));
    LLDB_REGISTER_METHOD_CONST(bool, SBType, IsValid, ());
    LLDB_REGISTER_METHOD(uint64_t, SBType, GetByteSize, ());
    LLDB_REGISTER_METHOD(bool, SBType, IsPointerType, ());
    LLDB_REGISTER_METHOD(bool, SBType, IsArrayType, ());
    LLDB_REGISTER_METHOD(bool, SBType, IsVectorType, ());
    LLDB_REGISTER_METHOD(bool, SBType, IsReferenceType, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetPointerType, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetPointeeType, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetReferenceType, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetTypedefedType, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetDereferencedType, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetArrayElementType, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetArrayType, (uint64_t));
    LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetVectorElementType, ());
    LLDB_REGISTER_METHOD(bool, SBType, IsFunctionType, ());
    LLDB_REGISTER_METHOD(bool, SBType, IsPolymorphicClass, ());
    LLDB_REGISTER_METHOD(bool, SBType, IsTypedefType, ());
    LLDB_REGISTER_METHOD(bool, SBType, IsAnonymousType, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetFunctionReturnType, ());
    LLDB_REGISTER_METHOD(lldb::SBTypeList, SBType, GetFunctionArgumentTypes,
                         ());
    LLDB_REGISTER_METHOD(uint32_t, SBType, GetNumberOfMemberFunctions, ());
    LLDB_REGISTER_METHOD(lldb::SBTypeMemberFunction, SBType,
                         GetMemberFunctionAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetUnqualifiedType, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetCanonicalType, ());
    LLDB_REGISTER_METHOD(lldb::BasicType, SBType, GetBasicType, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetBasicType, (lldb::BasicType));
    LLDB_REGISTER_METHOD(uint32_t, SBType, GetNumberOfDirectBaseClasses, ());
    LLDB_REGISTER_METHOD(uint32_t, SBType, GetNumberOfVirtualBaseClasses, ());
    LLDB_REGISTER_METHOD(uint32_t, SBType, GetNumberOfFields, ());
    LLDB_REGISTER_METHOD(bool, SBType, GetDescription,
                         (lldb::SBStream &, lldb::DescriptionLevel));
    LLDB_REGISTER_METHOD(lldb::SBTypeMember, SBType, GetDirectBaseClassAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBTypeMember, SBType, GetVirtualBaseClassAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBTypeEnumMemberList, SBType, GetEnumMembers,
                         ());
    LLDB_REGISTER_METHOD(lldb::SBTypeMember, SBType, GetFieldAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(bool, SBType, IsTypeComplete, ());
    LLDB_REGISTER_METHOD(uint32_t, SBType, GetTypeFlags, ());
    LLDB_REGISTER_METHOD(const char *, SBType, GetName, ());
    LLDB_REGISTER_METHOD(const char *, SBType, GetDisplayTypeName, ());
    LLDB_REGISTER_METHOD(lldb::TypeClass, SBType, GetTypeClass, ());
    LLDB_REGISTER_METHOD(uint32_t, SBType, GetNumberOfTemplateArguments, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBType, GetTemplateArgumentType,
                         (uint32_t));
    LLDB_REGISTER_METHOD(lldb::TemplateArgumentKind, SBType,
                         GetTemplateArgumentKind, (uint32_t));
    LLDB_REGISTER_CONSTRUCTOR(SBTypeList, ());
    LLDB_REGISTER_CONSTRUCTOR(SBTypeList, (const lldb::SBTypeList &));
    LLDB_REGISTER_METHOD(bool, SBTypeList, IsValid, ());
    LLDB_REGISTER_METHOD(lldb::SBTypeList &,
                         SBTypeList, operator=,(const lldb::SBTypeList &));
    LLDB_REGISTER_METHOD(void, SBTypeList, Append, (lldb::SBType));
    LLDB_REGISTER_METHOD(lldb::SBType, SBTypeList, GetTypeAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(uint32_t, SBTypeList, GetSize, ());
    LLDB_REGISTER_CONSTRUCTOR(SBTypeMember, ());
    LLDB_REGISTER_CONSTRUCTOR(SBTypeMember, (const lldb::SBTypeMember &));
    LLDB_REGISTER_METHOD(lldb::SBTypeMember &,
                         SBTypeMember, operator=,(const lldb::SBTypeMember &));
    LLDB_REGISTER_METHOD_CONST(bool, SBTypeMember, IsValid, ());
    LLDB_REGISTER_METHOD(const char *, SBTypeMember, GetName, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBTypeMember, GetType, ());
    LLDB_REGISTER_METHOD(uint64_t, SBTypeMember, GetOffsetInBytes, ());
    LLDB_REGISTER_METHOD(uint64_t, SBTypeMember, GetOffsetInBits, ());
    LLDB_REGISTER_METHOD(bool, SBTypeMember, IsBitfield, ());
    LLDB_REGISTER_METHOD(uint32_t, SBTypeMember, GetBitfieldSizeInBits, ());
    LLDB_REGISTER_METHOD(bool, SBTypeMember, GetDescription,
                         (lldb::SBStream &, lldb::DescriptionLevel));
    LLDB_REGISTER_CONSTRUCTOR(SBTypeMemberFunction, ());
    LLDB_REGISTER_CONSTRUCTOR(SBTypeMemberFunction,
                              (const lldb::SBTypeMemberFunction &));
    LLDB_REGISTER_METHOD(
        lldb::SBTypeMemberFunction &,
        SBTypeMemberFunction, operator=,(const lldb::SBTypeMemberFunction &));
    LLDB_REGISTER_METHOD_CONST(bool, SBTypeMemberFunction, IsValid, ());
    LLDB_REGISTER_METHOD(const char *, SBTypeMemberFunction, GetName, ());
    LLDB_REGISTER_METHOD(const char *, SBTypeMemberFunction, GetDemangledName,
                         ());
    LLDB_REGISTER_METHOD(const char *, SBTypeMemberFunction, GetMangledName,
                         ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBTypeMemberFunction, GetType, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBTypeMemberFunction, GetReturnType, ());
    LLDB_REGISTER_METHOD(uint32_t, SBTypeMemberFunction, GetNumberOfArguments,
                         ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBTypeMemberFunction,
                         GetArgumentTypeAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(lldb::MemberFunctionKind, SBTypeMemberFunction,
                         GetKind, ());
    LLDB_REGISTER_METHOD(bool, SBTypeMemberFunction, GetDescription,
                         (lldb::SBStream &, lldb::DescriptionLevel));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBTypeCategory, ());
    LLDB_REGISTER_CONSTRUCTOR(SBTypeCategory, (const lldb::SBTypeCategory &));
    LLDB_REGISTER_METHOD_CONST(bool, SBTypeCategory, IsValid, ());
    LLDB_REGISTER_METHOD(bool, SBTypeCategory, GetEnabled, ());
    LLDB_REGISTER_METHOD(void, SBTypeCategory, SetEnabled, (bool));
    LLDB_REGISTER_METHOD(const char *, SBTypeCategory, GetName, ());
    LLDB_REGISTER_METHOD(lldb::LanguageType, SBTypeCategory, GetLanguageAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(uint32_t, SBTypeCategory, GetNumLanguages, ());
    LLDB_REGISTER_METHOD(void, SBTypeCategory, AddLanguage,
                         (lldb::LanguageType));
    LLDB_REGISTER_METHOD(uint32_t, SBTypeCategory, GetNumFormats, ());
    LLDB_REGISTER_METHOD(uint32_t, SBTypeCategory, GetNumSummaries, ());
    LLDB_REGISTER_METHOD(uint32_t, SBTypeCategory, GetNumFilters, ());
    LLDB_REGISTER_METHOD(uint32_t, SBTypeCategory, GetNumSynthetics, ());
    LLDB_REGISTER_METHOD(lldb::SBTypeNameSpecifier, SBTypeCategory,
                         GetTypeNameSpecifierForFilterAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBTypeNameSpecifier, SBTypeCategory,
                         GetTypeNameSpecifierForFormatAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBTypeNameSpecifier, SBTypeCategory,
                         GetTypeNameSpecifierForSummaryAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBTypeNameSpecifier, SBTypeCategory,
                         GetTypeNameSpecifierForSyntheticAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBTypeFilter, SBTypeCategory, GetFilterForType,
                         (lldb::SBTypeNameSpecifier));
    LLDB_REGISTER_METHOD(lldb::SBTypeFormat, SBTypeCategory, GetFormatForType,
                         (lldb::SBTypeNameSpecifier));
    LLDB_REGISTER_METHOD(lldb::SBTypeSummary, SBTypeCategory, GetSummaryForType,
                         (lldb::SBTypeNameSpecifier));
    LLDB_REGISTER_METHOD(lldb::SBTypeSynthetic, SBTypeCategory,
                         GetSyntheticForType, (lldb::SBTypeNameSpecifier));
    LLDB_REGISTER_METHOD(lldb::SBTypeFilter, SBTypeCategory, GetFilterAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBTypeFormat, SBTypeCategory, GetFormatAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBTypeSummary, SBTypeCategory, GetSummaryAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBTypeSynthetic, SBTypeCategory,
                         GetSyntheticAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(bool, SBTypeCategory, AddTypeFormat,
                         (lldb::SBTypeNameSpecifier, lldb::SBTypeFormat));
    LLDB_REGISTER_METHOD(bool, SBTypeCategory, DeleteTypeFormat,
                         (lldb::SBTypeNameSpecifier));
    LLDB_REGISTER_METHOD(bool, SBTypeCategory, AddTypeSummary,
                         (lldb::SBTypeNameSpecifier, lldb::SBTypeSummary));
    LLDB_REGISTER_METHOD(bool, SBTypeCategory, DeleteTypeSummary,
                         (lldb::SBTypeNameSpecifier));
    LLDB_REGISTER_METHOD(bool, SBTypeCategory, AddTypeFilter,
                         (lldb::SBTypeNameSpecifier, lldb::SBTypeFilter));
    LLDB_REGISTER_METHOD(bool, SBTypeCategory, DeleteTypeFilter,
                         (lldb::SBTypeNameSpecifier));
    LLDB_REGISTER_METHOD(bool, SBTypeCategory, AddTypeSynthetic,
                         (lldb::SBTypeNameSpecifier, lldb::SBTypeSynthetic));
    LLDB_REGISTER_METHOD(bool, SBTypeCategory, DeleteTypeSynthetic,
                         (lldb::SBTypeNameSpecifier));
    LLDB_REGISTER_METHOD(bool, SBTypeCategory, GetDescription,
                         (lldb::SBStream &, lldb::DescriptionLevel));
    LLDB_REGISTER_METHOD(
        lldb::SBTypeCategory &,
        SBTypeCategory, operator=,(const lldb::SBTypeCategory &));
    LLDB_REGISTER_METHOD(bool,
                         SBTypeCategory, operator==,(lldb::SBTypeCategory &));
    LLDB_REGISTER_METHOD(bool,
                         SBTypeCategory, operator!=,(lldb::SBTypeCategory &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBTypeEnumMember, ());
    LLDB_REGISTER_CONSTRUCTOR(SBTypeEnumMember,
                              (const lldb::SBTypeEnumMember &));
    LLDB_REGISTER_METHOD(
        lldb::SBTypeEnumMember &,
        SBTypeEnumMember, operator=,(const lldb::SBTypeEnumMember &));
    LLDB_REGISTER_METHOD_CONST(bool, SBTypeEnumMember, IsValid, ());
    LLDB_REGISTER_METHOD(const char *, SBTypeEnumMember, GetName, ());
    LLDB_REGISTER_METHOD(int64_t, SBTypeEnumMember, GetValueAsSigned, ());
    LLDB_REGISTER_METHOD(uint64_t, SBTypeEnumMember, GetValueAsUnsigned, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBTypeEnumMember, GetType, ());
    LLDB_REGISTER_CONSTRUCTOR(SBTypeEnumMemberList, ());
    LLDB_REGISTER_CONSTRUCTOR(SBTypeEnumMemberList,
                              (const lldb::SBTypeEnumMemberList &));
    LLDB_REGISTER_METHOD(bool, SBTypeEnumMemberList, IsValid, ());
    LLDB_REGISTER_METHOD(
        lldb::SBTypeEnumMemberList &,
        SBTypeEnumMemberList, operator=,(const lldb::SBTypeEnumMemberList &));
    LLDB_REGISTER_METHOD(void, SBTypeEnumMemberList, Append,
                         (lldb::SBTypeEnumMember));
    LLDB_REGISTER_METHOD(lldb::SBTypeEnumMember, SBTypeEnumMemberList,
                         GetTypeEnumMemberAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(uint32_t, SBTypeEnumMemberList, GetSize, ());
    LLDB_REGISTER_METHOD(bool, SBTypeEnumMember, GetDescription,
                         (lldb::SBStream &, lldb::DescriptionLevel));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBTypeFilter, ());
    LLDB_REGISTER_CONSTRUCTOR(SBTypeFilter, (uint32_t));
    LLDB_REGISTER_CONSTRUCTOR(SBTypeFilter, (const lldb::SBTypeFilter &));
    LLDB_REGISTER_METHOD_CONST(bool, SBTypeFilter, IsValid, ());
    LLDB_REGISTER_METHOD(uint32_t, SBTypeFilter, GetOptions, ());
    LLDB_REGISTER_METHOD(void, SBTypeFilter, SetOptions, (uint32_t));
    LLDB_REGISTER_METHOD(bool, SBTypeFilter, GetDescription,
                         (lldb::SBStream &, lldb::DescriptionLevel));
    LLDB_REGISTER_METHOD(void, SBTypeFilter, Clear, ());
    LLDB_REGISTER_METHOD(uint32_t, SBTypeFilter, GetNumberOfExpressionPaths,
                         ());
    LLDB_REGISTER_METHOD(const char *, SBTypeFilter, GetExpressionPathAtIndex,
                         (uint32_t));
    LLDB_REGISTER_METHOD(bool, SBTypeFilter, ReplaceExpressionPathAtIndex,
                         (uint32_t, const char *));
    LLDB_REGISTER_METHOD(void, SBTypeFilter, AppendExpressionPath,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::SBTypeFilter &,
                         SBTypeFilter, operator=,(const lldb::SBTypeFilter &));
    LLDB_REGISTER_METHOD(bool, SBTypeFilter, operator==,(lldb::SBTypeFilter &));
    LLDB_REGISTER_METHOD(bool, SBTypeFilter, IsEqualTo, (lldb::SBTypeFilter &));
    LLDB_REGISTER_METHOD(bool, SBTypeFilter, operator!=,(lldb::SBTypeFilter &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBTypeFormat, ());
    LLDB_REGISTER_CONSTRUCTOR(SBTypeFormat, (lldb::Format, uint32_t));
    LLDB_REGISTER_CONSTRUCTOR(SBTypeFormat, (const char *, uint32_t));
    LLDB_REGISTER_CONSTRUCTOR(SBTypeFormat, (const lldb::SBTypeFormat &));
    LLDB_REGISTER_METHOD_CONST(bool, SBTypeFormat, IsValid, ());
    LLDB_REGISTER_METHOD(lldb::Format, SBTypeFormat, GetFormat, ());
    LLDB_REGISTER_METHOD(const char *, SBTypeFormat, GetTypeName, ());
    LLDB_REGISTER_METHOD(uint32_t, SBTypeFormat, GetOptions, ());
    LLDB_REGISTER_METHOD(void, SBTypeFormat, SetFormat, (lldb::Format));
    LLDB_REGISTER_METHOD(void, SBTypeFormat, SetTypeName, (const char *));
    LLDB_REGISTER_METHOD(void, SBTypeFormat, SetOptions, (uint32_t));
    LLDB_REGISTER_METHOD(bool, SBTypeFormat, GetDescription,
                         (lldb::SBStream &, lldb::DescriptionLevel));
    LLDB_REGISTER_METHOD(lldb::SBTypeFormat &,
                         SBTypeFormat, operator=,(const lldb::SBTypeFormat &));
    LLDB_REGISTER_METHOD(bool, SBTypeFormat, operator==,(lldb::SBTypeFormat &));
    LLDB_REGISTER_METHOD(bool, SBTypeFormat, IsEqualTo, (lldb::SBTypeFormat &));
    LLDB_REGISTER_METHOD(bool, SBTypeFormat, operator!=,(lldb::SBTypeFormat &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBTypeNameSpecifier, ());
    LLDB_REGISTER_CONSTRUCTOR(SBTypeNameSpecifier, (const char *, bool));
    LLDB_REGISTER_CONSTRUCTOR(SBTypeNameSpecifier, (lldb::SBType));
    LLDB_REGISTER_CONSTRUCTOR(SBTypeNameSpecifier,
                              (const lldb::SBTypeNameSpecifier &));
    LLDB_REGISTER_METHOD_CONST(bool, SBTypeNameSpecifier, IsValid, ());
    LLDB_REGISTER_METHOD(const char *, SBTypeNameSpecifier, GetName, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBTypeNameSpecifier, GetType, ());
    LLDB_REGISTER_METHOD(bool, SBTypeNameSpecifier, IsRegex, ());
    LLDB_REGISTER_METHOD(bool, SBTypeNameSpecifier, GetDescription,
                         (lldb::SBStream &, lldb::DescriptionLevel));
    LLDB_REGISTER_METHOD(
        lldb::SBTypeNameSpecifier &,
        SBTypeNameSpecifier, operator=,(const lldb::SBTypeNameSpecifier &));
    LLDB_REGISTER_METHOD(
        bool, SBTypeNameSpecifier, operator==,(lldb::SBTypeNameSpecifier &));
    LLDB_REGISTER_METHOD(bool, SBTypeNameSpecifier, IsEqualTo,
                         (lldb::SBTypeNameSpecifier &));
    LLDB_REGISTER_METHOD(
        bool, SBTypeNameSpecifier, operator!=,(lldb::SBTypeNameSpecifier &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBTypeSummaryOptions, ());
    LLDB_REGISTER_CONSTRUCTOR(SBTypeSummaryOptions,
                              (const lldb::SBTypeSummaryOptions &));
    LLDB_REGISTER_METHOD(bool, SBTypeSummaryOptions, IsValid, ());
    LLDB_REGISTER_METHOD(lldb::LanguageType, SBTypeSummaryOptions, GetLanguage,
                         ());
    LLDB_REGISTER_METHOD(lldb::TypeSummaryCapping, SBTypeSummaryOptions,
                         GetCapping, ());
    LLDB_REGISTER_METHOD(void, SBTypeSummaryOptions, SetLanguage,
                         (lldb::LanguageType));
    LLDB_REGISTER_METHOD(void, SBTypeSummaryOptions, SetCapping,
                         (lldb::TypeSummaryCapping));
    LLDB_REGISTER_CONSTRUCTOR(SBTypeSummaryOptions,
                              (const lldb_private::TypeSummaryOptions *));
    LLDB_REGISTER_CONSTRUCTOR(SBTypeSummary, ());
    LLDB_REGISTER_STATIC_METHOD(lldb::SBTypeSummary, SBTypeSummary,
                                CreateWithSummaryString,
                                (const char *, uint32_t));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBTypeSummary, SBTypeSummary,
                                CreateWithFunctionName,
                                (const char *, uint32_t));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBTypeSummary, SBTypeSummary,
                                CreateWithScriptCode, (const char *, uint32_t));
    LLDB_REGISTER_CONSTRUCTOR(SBTypeSummary, (const lldb::SBTypeSummary &));
    LLDB_REGISTER_METHOD_CONST(bool, SBTypeSummary, IsValid, ());
    LLDB_REGISTER_METHOD(bool, SBTypeSummary, IsFunctionCode, ());
    LLDB_REGISTER_METHOD(bool, SBTypeSummary, IsFunctionName, ());
    LLDB_REGISTER_METHOD(bool, SBTypeSummary, IsSummaryString, ());
    LLDB_REGISTER_METHOD(const char *, SBTypeSummary, GetData, ());
    LLDB_REGISTER_METHOD(uint32_t, SBTypeSummary, GetOptions, ());
    LLDB_REGISTER_METHOD(void, SBTypeSummary, SetOptions, (uint32_t));
    LLDB_REGISTER_METHOD(void, SBTypeSummary, SetSummaryString, (const char *));
    LLDB_REGISTER_METHOD(void, SBTypeSummary, SetFunctionName, (const char *));
    LLDB_REGISTER_METHOD(void, SBTypeSummary, SetFunctionCode, (const char *));
    LLDB_REGISTER_METHOD(bool, SBTypeSummary, GetDescription,
                         (lldb::SBStream &, lldb::DescriptionLevel));
    LLDB_REGISTER_METHOD(bool, SBTypeSummary, DoesPrintValue, (lldb::SBValue));
    LLDB_REGISTER_METHOD(
        lldb::SBTypeSummary &,
        SBTypeSummary, operator=,(const lldb::SBTypeSummary &));
    LLDB_REGISTER_METHOD(bool,
                         SBTypeSummary, operator==,(lldb::SBTypeSummary &));
    LLDB_REGISTER_METHOD(bool, SBTypeSummary, IsEqualTo,
                         (lldb::SBTypeSummary &));
    LLDB_REGISTER_METHOD(bool,
                         SBTypeSummary, operator!=,(lldb::SBTypeSummary &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBTypeSynthetic, ());
    LLDB_REGISTER_STATIC_METHOD(lldb::SBTypeSynthetic, SBTypeSynthetic,
                                CreateWithClassName, (const char *, uint32_t));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBTypeSynthetic, SBTypeSynthetic,
                                CreateWithScriptCode, (const char *, uint32_t));
    LLDB_REGISTER_CONSTRUCTOR(SBTypeSynthetic, (const lldb::SBTypeSynthetic &));
    LLDB_REGISTER_METHOD_CONST(bool, SBTypeSynthetic, IsValid, ());
    LLDB_REGISTER_METHOD(bool, SBTypeSynthetic, IsClassCode, ());
    LLDB_REGISTER_METHOD(bool, SBTypeSynthetic, IsClassName, ());
    LLDB_REGISTER_METHOD(const char *, SBTypeSynthetic, GetData, ());
    LLDB_REGISTER_METHOD(void, SBTypeSynthetic, SetClassName, (const char *));
    LLDB_REGISTER_METHOD(void, SBTypeSynthetic, SetClassCode, (const char *));
    LLDB_REGISTER_METHOD(uint32_t, SBTypeSynthetic, GetOptions, ());
    LLDB_REGISTER_METHOD(void, SBTypeSynthetic, SetOptions, (uint32_t));
    LLDB_REGISTER_METHOD(bool, SBTypeSynthetic, GetDescription,
                         (lldb::SBStream &, lldb::DescriptionLevel));
    LLDB_REGISTER_METHOD(
        lldb::SBTypeSynthetic &,
        SBTypeSynthetic, operator=,(const lldb::SBTypeSynthetic &));
    LLDB_REGISTER_METHOD(bool,
                         SBTypeSynthetic, operator==,(lldb::SBTypeSynthetic &));
    LLDB_REGISTER_METHOD(bool, SBTypeSynthetic, IsEqualTo,
                         (lldb::SBTypeSynthetic &));
    LLDB_REGISTER_METHOD(bool,
                         SBTypeSynthetic, operator!=,(lldb::SBTypeSynthetic &));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBUnixSignals, ());
    LLDB_REGISTER_CONSTRUCTOR(SBUnixSignals, (const lldb::SBUnixSignals &));
    LLDB_REGISTER_METHOD(
        const lldb::SBUnixSignals &,
        SBUnixSignals, operator=,(const lldb::SBUnixSignals &));
    LLDB_REGISTER_METHOD(void, SBUnixSignals, Clear, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBUnixSignals, IsValid, ());
    LLDB_REGISTER_METHOD_CONST(const char *, SBUnixSignals, GetSignalAsCString,
                               (int32_t));
    LLDB_REGISTER_METHOD_CONST(int32_t, SBUnixSignals, GetSignalNumberFromName,
                               (const char *));
    LLDB_REGISTER_METHOD_CONST(bool, SBUnixSignals, GetShouldSuppress,
                               (int32_t));
    LLDB_REGISTER_METHOD(bool, SBUnixSignals, SetShouldSuppress,
                         (int32_t, bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBUnixSignals, GetShouldStop, (int32_t));
    LLDB_REGISTER_METHOD(bool, SBUnixSignals, SetShouldStop, (int32_t, bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBUnixSignals, GetShouldNotify, (int32_t));
    LLDB_REGISTER_METHOD(bool, SBUnixSignals, SetShouldNotify, (int32_t, bool));
    LLDB_REGISTER_METHOD_CONST(int32_t, SBUnixSignals, GetNumSignals, ());
    LLDB_REGISTER_METHOD_CONST(int32_t, SBUnixSignals, GetSignalAtIndex,
                               (int32_t));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBValue, ());
    LLDB_REGISTER_CONSTRUCTOR(SBValue, (const lldb::ValueObjectSP &));
    LLDB_REGISTER_CONSTRUCTOR(SBValue, (const lldb::SBValue &));
    LLDB_REGISTER_METHOD(lldb::SBValue &,
                         SBValue, operator=,(const lldb::SBValue &));
    LLDB_REGISTER_METHOD(bool, SBValue, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBValue, Clear, ());
    LLDB_REGISTER_METHOD(lldb::SBError, SBValue, GetError, ());
    LLDB_REGISTER_METHOD(lldb::user_id_t, SBValue, GetID, ());
    LLDB_REGISTER_METHOD(const char *, SBValue, GetName, ());
    LLDB_REGISTER_METHOD(const char *, SBValue, GetTypeName, ());
    LLDB_REGISTER_METHOD(const char *, SBValue, GetDisplayTypeName, ());
    LLDB_REGISTER_METHOD(size_t, SBValue, GetByteSize, ());
    LLDB_REGISTER_METHOD(bool, SBValue, IsInScope, ());
    LLDB_REGISTER_METHOD(const char *, SBValue, GetValue, ());
    LLDB_REGISTER_METHOD(lldb::ValueType, SBValue, GetValueType, ());
    LLDB_REGISTER_METHOD(const char *, SBValue, GetObjectDescription, ());
    LLDB_REGISTER_METHOD(const char *, SBValue, GetTypeValidatorResult, ());
    LLDB_REGISTER_METHOD(lldb::SBType, SBValue, GetType, ());
    LLDB_REGISTER_METHOD(bool, SBValue, GetValueDidChange, ());
    LLDB_REGISTER_METHOD(const char *, SBValue, GetSummary, ());
    LLDB_REGISTER_METHOD(const char *, SBValue, GetSummary,
                         (lldb::SBStream &, lldb::SBTypeSummaryOptions &));
    LLDB_REGISTER_METHOD(const char *, SBValue, GetLocation, ());
    LLDB_REGISTER_METHOD(bool, SBValue, SetValueFromCString, (const char *));
    LLDB_REGISTER_METHOD(bool, SBValue, SetValueFromCString,
                         (const char *, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBTypeFormat, SBValue, GetTypeFormat, ());
    LLDB_REGISTER_METHOD(lldb::SBTypeSummary, SBValue, GetTypeSummary, ());
    LLDB_REGISTER_METHOD(lldb::SBTypeFilter, SBValue, GetTypeFilter, ());
    LLDB_REGISTER_METHOD(lldb::SBTypeSynthetic, SBValue, GetTypeSynthetic, ());
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, CreateChildAtOffset,
                         (const char *, uint32_t, lldb::SBType));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, Cast, (lldb::SBType));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, CreateValueFromExpression,
                         (const char *, const char *));
    LLDB_REGISTER_METHOD(
        lldb::SBValue, SBValue, CreateValueFromExpression,
        (const char *, const char *, lldb::SBExpressionOptions &));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, CreateValueFromAddress,
                         (const char *, lldb::addr_t, lldb::SBType));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, CreateValueFromData,
                         (const char *, lldb::SBData, lldb::SBType));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetChildAtIndex, (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetChildAtIndex,
                         (uint32_t, lldb::DynamicValueType, bool));
    LLDB_REGISTER_METHOD(uint32_t, SBValue, GetIndexOfChildWithName,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetChildMemberWithName,
                         (const char *));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetChildMemberWithName,
                         (const char *, lldb::DynamicValueType));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetDynamicValue,
                         (lldb::DynamicValueType));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetStaticValue, ());
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetNonSyntheticValue, ());
    LLDB_REGISTER_METHOD(lldb::DynamicValueType, SBValue, GetPreferDynamicValue,
                         ());
    LLDB_REGISTER_METHOD(void, SBValue, SetPreferDynamicValue,
                         (lldb::DynamicValueType));
    LLDB_REGISTER_METHOD(bool, SBValue, GetPreferSyntheticValue, ());
    LLDB_REGISTER_METHOD(void, SBValue, SetPreferSyntheticValue, (bool));
    LLDB_REGISTER_METHOD(bool, SBValue, IsDynamic, ());
    LLDB_REGISTER_METHOD(bool, SBValue, IsSynthetic, ());
    LLDB_REGISTER_METHOD(bool, SBValue, IsSyntheticChildrenGenerated, ());
    LLDB_REGISTER_METHOD(void, SBValue, SetSyntheticChildrenGenerated, (bool));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetValueForExpressionPath,
                         (const char *));
    LLDB_REGISTER_METHOD(int64_t, SBValue, GetValueAsSigned,
                         (lldb::SBError &, int64_t));
    LLDB_REGISTER_METHOD(uint64_t, SBValue, GetValueAsUnsigned,
                         (lldb::SBError &, uint64_t));
    LLDB_REGISTER_METHOD(int64_t, SBValue, GetValueAsSigned, (int64_t));
    LLDB_REGISTER_METHOD(uint64_t, SBValue, GetValueAsUnsigned, (uint64_t));
    LLDB_REGISTER_METHOD(bool, SBValue, MightHaveChildren, ());
    LLDB_REGISTER_METHOD(bool, SBValue, IsRuntimeSupportValue, ());
    LLDB_REGISTER_METHOD(uint32_t, SBValue, GetNumChildren, ());
    LLDB_REGISTER_METHOD(uint32_t, SBValue, GetNumChildren, (uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, Dereference, ());
    LLDB_REGISTER_METHOD(bool, SBValue, TypeIsPointerType, ());
    LLDB_REGISTER_METHOD(void *, SBValue, GetOpaqueType, ());
    LLDB_REGISTER_METHOD(lldb::SBTarget, SBValue, GetTarget, ());
    LLDB_REGISTER_METHOD(lldb::SBProcess, SBValue, GetProcess, ());
    LLDB_REGISTER_METHOD(lldb::SBThread, SBValue, GetThread, ());
    LLDB_REGISTER_METHOD(lldb::SBFrame, SBValue, GetFrame, ());
    LLDB_REGISTER_METHOD_CONST(lldb::ValueObjectSP, SBValue, GetSP, ());
    LLDB_REGISTER_METHOD(bool, SBValue, GetExpressionPath, (lldb::SBStream &));
    LLDB_REGISTER_METHOD(bool, SBValue, GetExpressionPath,
                         (lldb::SBStream &, bool));
    LLDB_REGISTER_METHOD_CONST(lldb::SBValue, SBValue, EvaluateExpression,
                               (const char *));
    LLDB_REGISTER_METHOD_CONST(
        lldb::SBValue, SBValue, EvaluateExpression,
        (const char *, const lldb::SBExpressionOptions &));
    LLDB_REGISTER_METHOD_CONST(
        lldb::SBValue, SBValue, EvaluateExpression,
        (const char *, const lldb::SBExpressionOptions &, const char *));
    LLDB_REGISTER_METHOD(bool, SBValue, GetDescription, (lldb::SBStream &));
    LLDB_REGISTER_METHOD(lldb::Format, SBValue, GetFormat, ());
    LLDB_REGISTER_METHOD(void, SBValue, SetFormat, (lldb::Format));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, AddressOf, ());
    LLDB_REGISTER_METHOD(lldb::addr_t, SBValue, GetLoadAddress, ());
    LLDB_REGISTER_METHOD(lldb::SBAddress, SBValue, GetAddress, ());
    LLDB_REGISTER_METHOD(lldb::SBData, SBValue, GetPointeeData,
                         (uint32_t, uint32_t));
    LLDB_REGISTER_METHOD(lldb::SBData, SBValue, GetData, ());
    LLDB_REGISTER_METHOD(bool, SBValue, SetData,
                         (lldb::SBData &, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBDeclaration, SBValue, GetDeclaration, ());
    LLDB_REGISTER_METHOD(lldb::SBWatchpoint, SBValue, Watch,
                         (bool, bool, bool, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBWatchpoint, SBValue, Watch,
                         (bool, bool, bool));
    LLDB_REGISTER_METHOD(lldb::SBWatchpoint, SBValue, WatchPointee,
                         (bool, bool, bool, lldb::SBError &));
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, Persist, ());
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBValueList, ());
    LLDB_REGISTER_CONSTRUCTOR(SBValueList, (const lldb::SBValueList &));
    LLDB_REGISTER_METHOD_CONST(bool, SBValueList, IsValid, ());
    LLDB_REGISTER_METHOD(void, SBValueList, Clear, ());
    LLDB_REGISTER_METHOD(const lldb::SBValueList &,
                         SBValueList, operator=,(const lldb::SBValueList &));
    LLDB_REGISTER_METHOD(void, SBValueList, Append, (const lldb::SBValue &));
    LLDB_REGISTER_METHOD(void, SBValueList, Append,
                         (const lldb::SBValueList &));
    LLDB_REGISTER_METHOD_CONST(lldb::SBValue, SBValueList, GetValueAtIndex,
                               (uint32_t));
    LLDB_REGISTER_METHOD_CONST(uint32_t, SBValueList, GetSize, ());
    LLDB_REGISTER_METHOD(lldb::SBValue, SBValueList, FindValueObjectByUID,
                         (lldb::user_id_t));
    LLDB_REGISTER_METHOD_CONST(lldb::SBValue, SBValueList, GetFirstValueByName,
                               (const char *));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBVariablesOptions, ());
    LLDB_REGISTER_CONSTRUCTOR(SBVariablesOptions,
                              (const lldb::SBVariablesOptions &));
    LLDB_REGISTER_METHOD(
        lldb::SBVariablesOptions &,
        SBVariablesOptions, operator=,(const lldb::SBVariablesOptions &));
    LLDB_REGISTER_METHOD_CONST(bool, SBVariablesOptions, IsValid, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBVariablesOptions, GetIncludeArguments,
                               ());
    LLDB_REGISTER_METHOD(void, SBVariablesOptions, SetIncludeArguments, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBVariablesOptions,
                               GetIncludeRecognizedArguments,
                               (const lldb::SBTarget &));
    LLDB_REGISTER_METHOD(void, SBVariablesOptions,
                         SetIncludeRecognizedArguments, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBVariablesOptions, GetIncludeLocals, ());
    LLDB_REGISTER_METHOD(void, SBVariablesOptions, SetIncludeLocals, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBVariablesOptions, GetIncludeStatics, ());
    LLDB_REGISTER_METHOD(void, SBVariablesOptions, SetIncludeStatics, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBVariablesOptions, GetInScopeOnly, ());
    LLDB_REGISTER_METHOD(void, SBVariablesOptions, SetInScopeOnly, (bool));
    LLDB_REGISTER_METHOD_CONST(bool, SBVariablesOptions,
                               GetIncludeRuntimeSupportValues, ());
    LLDB_REGISTER_METHOD(void, SBVariablesOptions,
                         SetIncludeRuntimeSupportValues, (bool));
    LLDB_REGISTER_METHOD_CONST(lldb::DynamicValueType, SBVariablesOptions,
                               GetUseDynamic, ());
    LLDB_REGISTER_METHOD(void, SBVariablesOptions, SetUseDynamic,
                         (lldb::DynamicValueType));
  }
  {
    LLDB_REGISTER_CONSTRUCTOR(SBWatchpoint, ());
    LLDB_REGISTER_CONSTRUCTOR(SBWatchpoint, (const lldb::WatchpointSP &));
    LLDB_REGISTER_CONSTRUCTOR(SBWatchpoint, (const lldb::SBWatchpoint &));
    LLDB_REGISTER_METHOD(const lldb::SBWatchpoint &,
                         SBWatchpoint, operator=,(const lldb::SBWatchpoint &));
    LLDB_REGISTER_METHOD(lldb::watch_id_t, SBWatchpoint, GetID, ());
    LLDB_REGISTER_METHOD_CONST(bool, SBWatchpoint, IsValid, ());
    LLDB_REGISTER_METHOD(lldb::SBError, SBWatchpoint, GetError, ());
    LLDB_REGISTER_METHOD(int32_t, SBWatchpoint, GetHardwareIndex, ());
    LLDB_REGISTER_METHOD(lldb::addr_t, SBWatchpoint, GetWatchAddress, ());
    LLDB_REGISTER_METHOD(size_t, SBWatchpoint, GetWatchSize, ());
    LLDB_REGISTER_METHOD(void, SBWatchpoint, SetEnabled, (bool));
    LLDB_REGISTER_METHOD(bool, SBWatchpoint, IsEnabled, ());
    LLDB_REGISTER_METHOD(uint32_t, SBWatchpoint, GetHitCount, ());
    LLDB_REGISTER_METHOD(uint32_t, SBWatchpoint, GetIgnoreCount, ());
    LLDB_REGISTER_METHOD(void, SBWatchpoint, SetIgnoreCount, (uint32_t));
    LLDB_REGISTER_METHOD(const char *, SBWatchpoint, GetCondition, ());
    LLDB_REGISTER_METHOD(void, SBWatchpoint, SetCondition, (const char *));
    LLDB_REGISTER_METHOD(bool, SBWatchpoint, GetDescription,
                         (lldb::SBStream &, lldb::DescriptionLevel));
    LLDB_REGISTER_METHOD(void, SBWatchpoint, Clear, ());
    LLDB_REGISTER_METHOD_CONST(lldb::WatchpointSP, SBWatchpoint, GetSP, ());
    LLDB_REGISTER_METHOD(void, SBWatchpoint, SetSP,
                         (const lldb::WatchpointSP &));
    LLDB_REGISTER_STATIC_METHOD(bool, SBWatchpoint, EventIsWatchpointEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(lldb::WatchpointEventType, SBWatchpoint,
                                GetWatchpointEventTypeFromEvent,
                                (const lldb::SBEvent &));
    LLDB_REGISTER_STATIC_METHOD(lldb::SBWatchpoint, SBWatchpoint,
                                GetWatchpointFromEvent,
                                (const lldb::SBEvent &));
  }
}

const char *SBReproducer::Capture(const char *path) {
  static std::string error;
  if (auto e =
          Reproducer::Initialize(ReproducerMode::Capture, FileSpec(path))) {
    error = llvm::toString(std::move(e));
    return error.c_str();
  }
  return nullptr;
}

const char *SBReproducer::Replay(const char *path) {
  static std::string error;
  if (auto e = Reproducer::Initialize(ReproducerMode::Replay, FileSpec(path))) {
    error = llvm::toString(std::move(e));
    return error.c_str();
  }

  repro::Loader *loader = repro::Reproducer::Instance().GetLoader();
  if (!loader) {
    error = "unable to get replay loader.";
    return error.c_str();
  }

  FileSpec file = loader->GetFile<SBInfo>();
  if (!file) {
    error = "unable to get replay data from reproducer.";
    return error.c_str();
  }

  SBRegistry registry;
  registry.Replay(file);

  return nullptr;
}

char lldb_private::repro::SBProvider::ID = 0;
const char *SBInfo::name = "sbapi";
const char *SBInfo::file = "sbapi.bin";
