//===-- LanguageRuntime.h ---------------------------------------------------*-
// C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_LanguageRuntime_h_
#define liblldb_LanguageRuntime_h_

#include "lldb/Breakpoint/BreakpointResolver.h"
#include "lldb/Breakpoint/BreakpointResolverName.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Expression/LLVMUserExpression.h"
#include "lldb/Symbol/DeclVendor.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/lldb-private.h"
#include "lldb/lldb-public.h"

namespace lldb_private {

class ExceptionSearchFilter : public SearchFilter {
public:
  ExceptionSearchFilter(const lldb::TargetSP &target_sp,
                        lldb::LanguageType language,
                        bool update_module_list = true);

  ~ExceptionSearchFilter() override = default;

  bool ModulePasses(const lldb::ModuleSP &module_sp) override;

  bool ModulePasses(const FileSpec &spec) override;

  void Search(Searcher &searcher) override;

  void GetDescription(Stream *s) override;

  static SearchFilter *
  CreateFromStructuredData(Target &target,
                           const StructuredData::Dictionary &data_dict,
                           Status &error);

  StructuredData::ObjectSP SerializeToStructuredData() override;

protected:
  lldb::LanguageType m_language;
  LanguageRuntime *m_language_runtime;
  lldb::SearchFilterSP m_filter_sp;

  lldb::SearchFilterSP DoCopyForBreakpoint(Breakpoint &breakpoint) override;

  void UpdateModuleListIfNeeded();
};

class LanguageRuntime : public PluginInterface {
public:
  ~LanguageRuntime() override;

  static LanguageRuntime *FindPlugin(Process *process,
                                     lldb::LanguageType language);

  static void InitializeCommands(CommandObject *parent);

  virtual lldb::LanguageType GetLanguageType() const = 0;

  virtual bool GetObjectDescription(Stream &str, ValueObject &object) = 0;

  virtual bool GetObjectDescription(Stream &str, Value &value,
                                    ExecutionContextScope *exe_scope) = 0;

  // this call should return true if it could set the name and/or the type
  virtual bool GetDynamicTypeAndAddress(ValueObject &in_value,
                                        lldb::DynamicValueType use_dynamic,
                                        TypeAndOrName &class_type_or_name,
                                        Address &address,
                                        Value::ValueType &value_type) = 0;

  // This call should return a CompilerType given a generic type name and an
  // ExecutionContextScope in which one can actually fetch any specialization
  // information required.
  virtual CompilerType GetConcreteType(ExecutionContextScope *exe_scope,
                                       ConstString abstract_type_name) {
    return CompilerType();
  }

  // This should be a fast test to determine whether it is likely that this
  // value would have a dynamic type.
  virtual bool CouldHaveDynamicValue(ValueObject &in_value) = 0;

  // The contract for GetDynamicTypeAndAddress() is to return a "bare-bones"
  // dynamic type For instance, given a Base* pointer,
  // GetDynamicTypeAndAddress() will return the type of Derived, not Derived*.
  // The job of this API is to correct this misalignment between the static
  // type and the discovered dynamic type
  virtual TypeAndOrName FixUpDynamicType(const TypeAndOrName &type_and_or_name,
                                         ValueObject &static_value) = 0;

  virtual void SetExceptionBreakpoints() {}

  virtual void ClearExceptionBreakpoints() {}

  virtual bool ExceptionBreakpointsAreSet() { return false; }

  virtual bool ExceptionBreakpointsExplainStop(lldb::StopInfoSP stop_reason) {
    return false;
  }

  static lldb::BreakpointSP
  CreateExceptionBreakpoint(Target &target, lldb::LanguageType language,
                            bool catch_bp, bool throw_bp,
                            bool is_internal = false);

  static lldb::BreakpointPreconditionSP
  GetExceptionPrecondition(lldb::LanguageType language, bool throw_bp);

  virtual lldb::ValueObjectSP GetExceptionObjectForThread(
      lldb::ThreadSP thread_sp) {
    return lldb::ValueObjectSP();
  }

  virtual lldb::ThreadSP GetBacktraceThreadFromException(
      lldb::ValueObjectSP thread_sp) {
    return lldb::ThreadSP();
  }

  Process *GetProcess() { return m_process; }

  Target &GetTargetRef() { return m_process->GetTarget(); }

  virtual DeclVendor *GetDeclVendor() { return nullptr; }

  virtual lldb::BreakpointResolverSP
  CreateExceptionResolver(Breakpoint *bkpt, bool catch_bp, bool throw_bp) = 0;

  virtual lldb::SearchFilterSP CreateExceptionSearchFilter() {
    return m_process->GetTarget().GetSearchFilterForModule(nullptr);
  }

  virtual bool GetTypeBitSize(const CompilerType &compiler_type,
                              uint64_t &size) {
    return false;
  }

  virtual void SymbolsDidLoad(const ModuleList &module_list) { return; }

  virtual lldb::ThreadPlanSP GetStepThroughTrampolinePlan(Thread &thread,
                                                          bool stop_others) = 0;

  /// Identify whether a name is a runtime value that should not be hidden by
  /// from the user interface.
  virtual bool IsWhitelistedRuntimeValue(ConstString name) { return false; }

  virtual llvm::Optional<CompilerType> GetRuntimeType(CompilerType base_type) {
    return llvm::None;
  }

  virtual void ModulesDidLoad(const ModuleList &module_list) {}

  // Called by ClangExpressionParser::PrepareForExecution to query for any
  // custom LLVM IR passes that need to be run before an expression is
  // assembled and run.
  virtual bool GetIRPasses(LLVMUserExpression::IRPasses &custom_passes) {
    return false;
  }

  // Given the name of a runtime symbol (e.g. in Objective-C, an ivar offset
  // symbol), try to determine from the runtime what the value of that symbol
  // would be. Useful when the underlying binary is stripped.
  virtual lldb::addr_t LookupRuntimeSymbol(ConstString name) {
    return LLDB_INVALID_ADDRESS;
  }

  virtual bool isA(const void *ClassID) const { return ClassID == &ID; }
  static char ID;

protected:
  // Classes that inherit from LanguageRuntime can see and modify these

  LanguageRuntime(Process *process);
  Process *m_process;

private:
  DISALLOW_COPY_AND_ASSIGN(LanguageRuntime);
};

} // namespace lldb_private

#endif // liblldb_LanguageRuntime_h_
