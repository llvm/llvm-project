//===-- CPPLanguageRuntime.h
//---------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CPPLanguageRuntime_h_
#define liblldb_CPPLanguageRuntime_h_

#include <vector>
#include "lldb/Core/PluginInterface.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class CPPLanguageRuntime : public LanguageRuntime {
public:
  enum class LibCppStdFunctionCallableCase {
    Lambda = 0,
    CallableObject,
    FreeOrMemberFunction,
    Invalid
  };

  struct LibCppStdFunctionCallableInfo {
    Symbol callable_symbol;
    Address callable_address;
    LineEntry callable_line_entry;
    lldb::addr_t member__f_pointer_value = 0u;
    LibCppStdFunctionCallableCase callable_case =
        LibCppStdFunctionCallableCase::Invalid;
  };

  LibCppStdFunctionCallableInfo
  FindLibCppStdFunctionCallableInfo(lldb::ValueObjectSP &valobj_sp);

  ~CPPLanguageRuntime() override;

  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeC_plus_plus;
  }

  virtual bool IsVTableName(const char *name) = 0;

  bool GetObjectDescription(Stream &str, ValueObject &object) override;

  bool GetObjectDescription(Stream &str, Value &value,
                            ExecutionContextScope *exe_scope) override;

  /// Obtain a ThreadPlan to get us into C++ constructs such as std::function.
  ///
  /// @param[in] thread
  ///     Curent thrad of execution.
  ///
  /// @param[in] stop_others
  ///     True if other threads should pause during execution.
  ///
  /// @return
  ///      A ThreadPlan Shared pointer
  lldb::ThreadPlanSP GetStepThroughTrampolinePlan(Thread &thread,
                                                  bool stop_others);

protected:
  //------------------------------------------------------------------
  // Classes that inherit from CPPLanguageRuntime can see and modify these
  //------------------------------------------------------------------
  CPPLanguageRuntime(Process *process);

private:
  DISALLOW_COPY_AND_ASSIGN(CPPLanguageRuntime);
};

} // namespace lldb_private

#endif // liblldb_CPPLanguageRuntime_h_
