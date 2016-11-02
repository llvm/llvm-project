//===-- OptionValueEnumeration.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionValueEnumeration_h_
#define liblldb_OptionValueEnumeration_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Interpreter/OptionValue.h"

namespace lldb_private {

class OptionValueEnumeration : public OptionValue {
public:
  typedef int64_t enum_type;
  struct EnumeratorInfo {
    enum_type value;
    const char *description;
  };
  typedef UniqueCStringMap<EnumeratorInfo> EnumerationMap;
  typedef EnumerationMap::Entry EnumerationMapEntry;

  OptionValueEnumeration(const OptionEnumValueElement *enumerators,
                         enum_type value);

  ~OptionValueEnumeration() override;

  //---------------------------------------------------------------------
  // Virtual subclass pure virtual overrides
  //---------------------------------------------------------------------

  OptionValue::Type GetType() const override { return eTypeEnum; }

  void DumpValue(const ExecutionContext *exe_ctx, Stream &strm,
                 uint32_t dump_mask) override;

  Error
  SetValueFromString(llvm::StringRef value,
                     VarSetOperationType op = eVarSetOperationAssign) override;

  bool Clear() override {
    m_current_value = m_default_value;
    m_value_was_set = false;
    return true;
  }

  lldb::OptionValueSP DeepCopy() const override;

  size_t AutoComplete(CommandInterpreter &interpreter, const char *s,
                      int match_start_point, int max_return_elements,
                      bool &word_complete, StringList &matches) override;

  //---------------------------------------------------------------------
  // Subclass specific functions
  //---------------------------------------------------------------------

  enum_type operator=(enum_type value) {
    m_current_value = value;
    return m_current_value;
  }

  enum_type GetCurrentValue() const { return m_current_value; }

  enum_type GetDefaultValue() const { return m_default_value; }

  void SetCurrentValue(enum_type value) { m_current_value = value; }

  void SetDefaultValue(enum_type value) { m_default_value = value; }

protected:
  void SetEnumerations(const OptionEnumValueElement *enumerators);

  enum_type m_current_value;
  enum_type m_default_value;
  EnumerationMap m_enumerations;
};

} // namespace lldb_private

#endif // liblldb_OptionValueEnumeration_h_
