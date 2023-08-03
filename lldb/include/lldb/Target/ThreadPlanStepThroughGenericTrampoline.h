//===-- ThreadPlanStepInRange.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_THREADPLANSTEPTHROUGHGENERICTRAMPOLINE_H
#define LLDB_TARGET_THREADPLANSTEPTHROUGHGENERICTRAMPOLINE_H

#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanShouldStopHere.h"
#include "lldb/Target/ThreadPlanStepRange.h"

namespace lldb_private {

class ThreadPlanStepThroughGenericTrampoline : public ThreadPlanStepRange,
                                               public ThreadPlanShouldStopHere {
public:
  ThreadPlanStepThroughGenericTrampoline(Thread &thread,
                                         lldb::RunMode stop_others);

  ~ThreadPlanStepThroughGenericTrampoline() override;

  void GetDescription(Stream *s, lldb::DescriptionLevel level) override;

  bool ShouldStop(Event *event_ptr) override;
  bool ValidatePlan(Stream *error) override;

protected:
  void SetFlagsToDefault() override {
    GetFlags().Set(
        ThreadPlanStepThroughGenericTrampoline::s_default_flag_values);
  }

private:
  // Need an appropriate marker for the current stack so we can tell step out
  // from step in.

  static uint32_t
      s_default_flag_values; // These are the default flag values
                             // for the ThreadPlanStepThroughGenericTrampoline.
  ThreadPlanStepThroughGenericTrampoline(
      const ThreadPlanStepThroughGenericTrampoline &) = delete;
  const ThreadPlanStepThroughGenericTrampoline &
  operator=(const ThreadPlanStepThroughGenericTrampoline &) = delete;
};

} // namespace lldb_private

#endif // LLDB_TARGET_THREADPLANSTEPTHROUGHGENERICTRAMPOLINE_H
