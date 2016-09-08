//===-- OptionGroupBoolean.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionGroupBoolean_h_
#define liblldb_OptionGroupBoolean_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/OptionValueBoolean.h"
#include "lldb/Interpreter/Options.h"

namespace lldb_private {
//-------------------------------------------------------------------------
// OptionGroupBoolean
//-------------------------------------------------------------------------

class OptionGroupBoolean : public OptionGroup {
public:
  // When 'no_argument_toggle_default' is true, then setting the option
  // value does NOT require an argument, it sets the boolean value to the
  // inverse of the default value
  OptionGroupBoolean(uint32_t usage_mask, bool required,
                     const char *long_option, int short_option,
                     const char *usage_text, bool default_value,
                     bool no_argument_toggle_default);

  ~OptionGroupBoolean() override;

  uint32_t GetNumDefinitions() override { return 1; }

  const OptionDefinition *GetDefinitions() override {
    return &m_option_definition;
  }

  Error SetOptionValue(CommandInterpreter &interpreter, uint32_t option_idx,
                       const char *option_value) override;

  void OptionParsingStarting(CommandInterpreter &interpreter) override;

  OptionValueBoolean &GetOptionValue() { return m_value; }

  const OptionValueBoolean &GetOptionValue() const { return m_value; }

protected:
  OptionValueBoolean m_value;
  OptionDefinition m_option_definition;
};

} // namespace lldb_private

#endif // liblldb_OptionGroupBoolean_h_
