//===-- OptionGroupPythonClassWithDict.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_OPTIONGROUPPYTHONCLASSWITHDICT_H
#define LLDB_INTERPRETER_OPTIONGROUPPYTHONCLASSWITHDICT_H

#include "lldb/lldb-types.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Utility/StructuredData.h"

namespace lldb_private {

// Use this Option group if you have a python class that implements some
// Python extension point, and you pass a SBStructuredData to the class 
// __init__ method.  
// class_option specifies the class name
// the key and value options are read in in pairs, and a 
// StructuredData::Dictionary is constructed with those pairs.
class OptionGroupPythonClassWithDict : public OptionGroup {
public:
  OptionGroupPythonClassWithDict(const char *class_use,
                                 bool is_class = true,
                                 int class_option = 'C',
                                 int key_option = 'k', 
                                 int value_option = 'v');
                      
  ~OptionGroupPythonClassWithDict() override;

  llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
    return llvm::ArrayRef<OptionDefinition>(m_option_definition);
  }

  Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_value,
                        ExecutionContext *execution_context) override;

  void OptionParsingStarting(ExecutionContext *execution_context) override;
  Status OptionParsingFinished(ExecutionContext *execution_context) override;
  
  const StructuredData::DictionarySP GetStructuredData() {
    return m_dict_sp;
  }
  const std::string &GetName() {
    return m_name;
  }

protected:
  std::string m_name;
  std::string m_current_key;
  StructuredData::DictionarySP m_dict_sp;
  std::string m_class_usage_text, m_key_usage_text, m_value_usage_text;
  bool m_is_class;
  OptionDefinition m_option_definition[4];
};

} // namespace lldb_private

#endif // LLDB_INTERPRETER_OPTIONGROUPPYTHONCLASSWITHDICT_H
