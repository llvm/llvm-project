/* TO_UPSTREAM(BoundsSafety) ON */
//===-- BoundsSafetyTrapFrameRecognizer.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_BoundsSafetyTrapFrameRegognizer_h_
#define liblldb_BoundsSafetyTrapFrameRegognizer_h_

#include "lldb/Target/StackFrameRecognizer.h"

namespace lldb_private {

void RegisterBoundsSafetyTrapFrameRecognizer(Process &process);

/// Holds the stack frame that caused the BoundsSafety trap and the inlined stop
/// reason message.
class BoundsSafetyTrapRecognizedStackFrame : public RecognizedStackFrame {
public:
  BoundsSafetyTrapRecognizedStackFrame(lldb::StackFrameSP most_relevant_frame_sp,
                                    llvm::StringRef stop_desc);
  lldb::StackFrameSP GetMostRelevantFrame() override;

private:
  lldb::StackFrameSP m_most_relevant_frame;
};

/// When a thread stops, it checks the current frame contains a BoundsSafety Trap
/// diagnostic. If so, it returns a \a
/// BoundsSafetyTrapRecognizedStackFrame holding the diagnostic a stop reason
/// description with and the parent frame as the most relavant frame.
class BoundsSafetyTrapFrameRecognizer : public StackFrameRecognizer {
public:
  std::string GetName() override {
    return "Pointer Check Trap StackFrame Recognizer";
  }
  lldb::RecognizedStackFrameSP
  RecognizeFrame(lldb::StackFrameSP frame) override;
};

} // namespace lldb_private

#endif // liblldb_BoundsSafetyTrapFrameRegognizer_h_
/* TO_UPSTREAM(BoundsSafety) OFF */