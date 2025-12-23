//===-- SBVariableAnnotator.h -----------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBVARIABLEANNOTATOR_H
#define LLDB_API_SBVARIABLEANNOTATOR_H

#include "lldb/API/SBDefines.h"

namespace lldb {

class LLDB_API SBVariableAnnotator {
public:
  SBVariableAnnotator();

  SBVariableAnnotator(const SBVariableAnnotator &rhs);

  const SBVariableAnnotator &operator=(const SBVariableAnnotator &rhs);

  ~SBVariableAnnotator();

  explicit operator bool() const;

  bool IsValid() const;

  /// Get variable annotations for this instruction as structured data.
  /// Returns an array of dictionaries, each containing:
  /// - "variable_name": string name of the variable
  /// - "location_description": string description of where variable is stored
  ///   ("RDI", "R15", "undef", etc.)
  /// - "is_live": boolean indicates if variable is live at this instruction
  /// - "start_address": unsigned integer address where this annotation becomes
  ///   valid
  /// - "end_address": unsigned integer address where this annotation becomes
  ///   invalid
  /// - "register_kind": unsigned integer indicating the register numbering
  /// scheme
  /// - "decl_file": string path to the file where variable is declared
  /// - "decl_line": unsigned integer line number where variable is declared
  /// - "type_name": string type name of the variable
  lldb::SBStructuredData AnnotateStructured(SBInstruction inst);

protected:
  SBVariableAnnotator(const lldb::VariableAnnotatorSP &annotator_sp);

  lldb::VariableAnnotatorSP GetSP() const;

  void SetSP(const lldb::VariableAnnotatorSP &annotator_sp);

private:
  lldb::VariableAnnotatorSP m_opaque_sp;
};
} // namespace lldb
#endif // LLDB_API_SBVARIABLEANNOTATOR_H