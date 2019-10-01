//===-- ThreadPlanTracer.h --------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanTracer_h_
#define liblldb_ThreadPlanTracer_h_

#include "lldb/Symbol/TaggedASTType.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class ThreadPlanTracer {
  friend class ThreadPlan;

public:
  enum ThreadPlanTracerStyle {
    eLocation = 0,
    eStateChange,
    eCheckFrames,
    ePython
  };

  ThreadPlanTracer(Thread &thread, lldb::StreamSP &stream_sp);
  ThreadPlanTracer(Thread &thread);

  virtual ~ThreadPlanTracer() = default;

  virtual void TracingStarted() {}

  virtual void TracingEnded() {}

  bool EnableTracing(bool value) {
    bool old_value = m_enabled;
    m_enabled = value;
    if (old_value == false && value == true)
      TracingStarted();
    else if (old_value == true && value == false)
      TracingEnded();

    return old_value;
  }

  bool TracingEnabled() { return m_enabled; }

  bool EnableSingleStep(bool value) {
    bool old_value = m_single_step;
    m_single_step = value;
    return old_value;
  }

  bool SingleStepEnabled() { return m_single_step; }

protected:
  Thread &m_thread;

  Stream *GetLogStream();

  virtual void Log();

private:
  bool TracerExplainsStop();

  bool m_single_step;
  bool m_enabled;
  lldb::StreamSP m_stream_sp;
};

class ThreadPlanAssemblyTracer : public ThreadPlanTracer {
public:
  ThreadPlanAssemblyTracer(Thread &thread, lldb::StreamSP &stream_sp);
  ThreadPlanAssemblyTracer(Thread &thread);
  ~ThreadPlanAssemblyTracer() override;

  void TracingStarted() override;
  void TracingEnded() override;
  void Log() override;

private:
  Disassembler *GetDisassembler();

  TypeFromUser GetIntPointerType();

  lldb::DisassemblerSP m_disassembler_sp;
  TypeFromUser m_intptr_type;
  std::vector<RegisterValue> m_register_values;
  lldb::DataBufferSP m_buffer_sp;
};

} // namespace lldb_private

#endif // liblldb_ThreadPlanTracer_h_
