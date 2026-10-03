// RUN: rm -rf %t && split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -fapinotes -Wapinotes -isystem %t/SystemHeaderAPINotes %t/test.cpp -x c++ 2>&1 | FileCheck %s

// CHECK: warning: API notes entry for 'systemHeaderUnmatched' has unmatched Where.Parameters [double]

//--- test.cpp
#include "SystemHeaderAPINotes.h"

//--- SystemHeaderAPINotes/SystemHeaderAPINotes.h
#ifndef SYSTEM_HEADER_API_NOTES_H
#define SYSTEM_HEADER_API_NOTES_H

#pragma clang system_header

void systemHeaderUnmatched(int);

#endif // SYSTEM_HEADER_API_NOTES_H

//--- SystemHeaderAPINotes/APINotes.apinotes
---
Name: SystemHeaderAPINotes
Functions:
- Name: systemHeaderUnmatched
  Where:
    Parameters:
    - double
  SwiftName: systemHeaderUnmatched(_:)
...
