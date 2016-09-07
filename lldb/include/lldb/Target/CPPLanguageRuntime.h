//===-- CPPLanguageRuntime.h
//---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CPPLanguageRuntime_h_
#define liblldb_CPPLanguageRuntime_h_

// C Includes
// C++ Includes
#include <vector>
// Other libraries and framework includes
// Project includes
#include "lldb/Core/PluginInterface.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class CPPLanguageRuntime : public LanguageRuntime {
public:
  ~CPPLanguageRuntime() override;

  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeC_plus_plus;
  }

  virtual bool IsVTableName(const char *name) = 0;

  bool GetObjectDescription(Stream &str, ValueObject &object) override;

  bool GetObjectDescription(Stream &str, Value &value,
                            ExecutionContextScope *exe_scope) override;

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
