//===-- StackFrameList.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StackFrameList_h_
#define liblldb_StackFrameList_h_

// C Includes
// C++ Includes
#include <memory>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Host/Mutex.h"
#include "lldb/Target/StackFrame.h"

namespace lldb_private {

class StackFrameList {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  StackFrameList(Thread &thread, const lldb::StackFrameListSP &prev_frames_sp,
                 bool show_inline_frames);

  ~StackFrameList();

  uint32_t GetNumFrames(bool can_create = true);

  lldb::StackFrameSP GetFrameAtIndex(uint32_t idx);

  lldb::StackFrameSP GetFrameWithConcreteFrameIndex(uint32_t unwind_idx);

  lldb::StackFrameSP GetFrameWithStackID(const StackID &stack_id);

  // Mark a stack frame as the current frame
  uint32_t SetSelectedFrame(lldb_private::StackFrame *frame);

  uint32_t GetSelectedFrameIndex() const;

  // Mark a stack frame as the current frame using the frame index
  bool SetSelectedFrameByIndex(uint32_t idx);

  uint32_t GetVisibleStackFrameIndex(uint32_t idx) {
    if (m_current_inlined_depth < UINT32_MAX)
      return idx - m_current_inlined_depth;
    else
      return idx;
  }

  void CalculateCurrentInlinedDepth();

  void SetDefaultFileAndLineToSelectedFrame();

  void Clear();

  void InvalidateFrames(uint32_t start_idx);

  void Dump(Stream *s);

  lldb::StackFrameSP
  GetStackFrameSPForStackFramePtr(StackFrame *stack_frame_ptr);

  size_t GetStatus(Stream &strm, uint32_t first_frame, uint32_t num_frames,
                   bool show_frame_info, uint32_t num_frames_with_source,
                   const char *frame_marker = nullptr);

protected:
  friend class Thread;

  bool SetFrameAtIndex(uint32_t idx, lldb::StackFrameSP &frame_sp);

  static void Merge(std::unique_ptr<StackFrameList> &curr_ap,
                    lldb::StackFrameListSP &prev_sp);

  void GetFramesUpTo(uint32_t end_idx);

  bool GetAllFramesFetched() { return m_concrete_frames_fetched == UINT32_MAX; }

  void SetAllFramesFetched() { m_concrete_frames_fetched = UINT32_MAX; }

  bool DecrementCurrentInlinedDepth();

  void ResetCurrentInlinedDepth();

  uint32_t GetCurrentInlinedDepth();

  void SetCurrentInlinedDepth(uint32_t new_depth);

  typedef std::vector<lldb::StackFrameSP> collection;
  typedef collection::iterator iterator;
  typedef collection::const_iterator const_iterator;

  Thread &m_thread;
  lldb::StackFrameListSP m_prev_frames_sp;
  mutable Mutex m_mutex;
  collection m_frames;
  uint32_t m_selected_frame_idx;
  uint32_t m_concrete_frames_fetched;
  uint32_t m_current_inlined_depth;
  lldb::addr_t m_current_inlined_pc;
  bool m_show_inlined_frames;

private:
  DISALLOW_COPY_AND_ASSIGN(StackFrameList);
};

} // namespace lldb_private

#endif // liblldb_StackFrameList_h_
