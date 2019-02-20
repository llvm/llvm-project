//===-- SBBreakpointOptionCommon.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBBreakpointOptionCommons_h_
#define LLDB_SBBreakpointOptionCommons_h_

#include "lldb/API/SBDefines.h"
#include "lldb/Utility/Baton.h"

namespace lldb
{
struct CallbackData {
  SBBreakpointHitCallback callback;
  void *callback_baton;
};

class SBBreakpointCallbackBaton : public lldb_private::TypedBaton<CallbackData> {
public:
  SBBreakpointCallbackBaton(SBBreakpointHitCallback callback,
                            void *baton);

  ~SBBreakpointCallbackBaton();

  static bool PrivateBreakpointHitCallback(void *baton,
                                           lldb_private::StoppointCallbackContext *ctx,
                                           lldb::user_id_t break_id,
                                           lldb::user_id_t break_loc_id);
};

} // namespace lldb
#endif // LLDB_SBBreakpointOptionCommons_h_
