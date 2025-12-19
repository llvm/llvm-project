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

protected:
  SBVariableAnnotator(const lldb::VariableAnnotatorSP &annotator_sp);

  lldb::VariableAnnotatorSP GetSP() const;

  void SetSP(const lldb::VariableAnnotatorSP &annotator_sp);

private:
  lldb::VariableAnnotatorSP m_opaque_sp;
};
} // namespace lldb
#endif // LLDB_API_SBVARIABLEANNOTATOR_H