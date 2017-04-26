//===-- SBStructuredData.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SBStructuredData_h
#define SBStructuredData_h

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBModule.h"

namespace lldb {

class SBStructuredData {
public:
  SBStructuredData();

  SBStructuredData(const lldb::SBStructuredData &rhs);

  SBStructuredData(const lldb::EventSP &event_sp);

  ~SBStructuredData();

  lldb::SBStructuredData &operator=(const lldb::SBStructuredData &rhs);

  bool IsValid() const;

  lldb::SBError SetFromJSON(lldb::SBStream &stream);

  void Clear();

  lldb::SBError GetAsJSON(lldb::SBStream &stream) const;

  lldb::SBError GetDescription(lldb::SBStream &stream) const;

protected:
  friend class SBTraceOptions;

  StructuredDataImplUP m_impl_up;
};
}

#endif /* SBStructuredData_h */
