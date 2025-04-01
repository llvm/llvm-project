//===-- StackFrameList.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_STACKFRAMELIST_H
#define LLDB_TARGET_STACKFRAMELIST_H

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "lldb/Target/StackFrame.h"

namespace lldb_private {

class ScriptedThread;

class StackFrameList {
public:
  // Constructors and Destructors
  StackFrameList(Thread &thread, const lldb::StackFrameListSP &prev_frames_sp,
                 bool show_inline_frames);

  ~StackFrameList();

  /// Get the number of visible frames. Frames may be created if \p can_create
  /// is true. Synthetic (inline) frames expanded from the concrete frame #0
  /// (aka invisible frames) are not included in this count.
  uint32_t GetNumFrames(bool can_create = true);

  /// Get the frame at index \p idx. Invisible frames cannot be indexed.
  lldb::StackFrameSP GetFrameAtIndex(uint32_t idx);

  /// Get the first concrete frame with index greater than or equal to \p idx.
  /// Unlike \ref GetFrameAtIndex, this cannot return a synthetic frame.
  lldb::StackFrameSP GetFrameWithConcreteFrameIndex(uint32_t unwind_idx);

  /// Retrieve the stack frame with the given ID \p stack_id.
  lldb::StackFrameSP GetFrameWithStackID(const StackID &stack_id);

  /// Mark a stack frame as the currently selected frame and return its index.
  uint32_t SetSelectedFrame(lldb_private::StackFrame *frame);

  /// Get the currently selected frame index.
  /// We should only call SelectMostRelevantFrame if (a) the user hasn't already
  /// selected a frame, and (b) if this really is a user facing
  /// "GetSelectedFrame".  SMRF runs the frame recognizers which can do
  /// arbitrary work that ends up being dangerous to do internally.  Also,
  /// for most internal uses we don't actually want the frame changed by the
  /// SMRF logic.  So unless this is in a command or SB API, you should
  /// pass false here.
  uint32_t
  GetSelectedFrameIndex(SelectMostRelevant select_most_relevant_frame);

  /// Mark a stack frame as the currently selected frame using the frame index
  /// \p idx. Like \ref GetFrameAtIndex, invisible frames cannot be selected.
  bool SetSelectedFrameByIndex(uint32_t idx);

  /// If the current inline depth (i.e the number of invisible frames) is valid,
  /// subtract it from \p idx. Otherwise simply return \p idx.
  uint32_t GetVisibleStackFrameIndex(uint32_t idx) {
    if (m_current_inlined_depth < UINT32_MAX)
      return idx - m_current_inlined_depth;
    else
      return idx;
  }

  /// Calculate and set the current inline depth. This may be used to update
  /// the StackFrameList's set of inline frames when execution stops, e.g when
  /// a breakpoint is hit.
  void CalculateCurrentInlinedDepth();

  /// If the currently selected frame comes from the currently selected thread,
  /// point the default file and line of the thread's target to the location
  /// specified by the frame.
  void SetDefaultFileAndLineToSelectedFrame();

  /// Clear the cache of frames.
  void Clear();

  void Dump(Stream *s);

  /// If \p stack_frame_ptr is contained in this StackFrameList, return its
  /// wrapping shared pointer.
  lldb::StackFrameSP
  GetStackFrameSPForStackFramePtr(StackFrame *stack_frame_ptr);

  size_t GetStatus(Stream &strm, uint32_t first_frame, uint32_t num_frames,
                   bool show_frame_info, uint32_t num_frames_with_source,
                   bool show_unique = false, bool show_hidden = false,
                   const char *frame_marker = nullptr);

  /// Returns whether we have currently fetched all the frames of a stack.
  bool WereAllFramesFetched() const;

protected:
  friend class Thread;
  friend class ScriptedThread;

  /// Use this API to build a stack frame list (used for scripted threads, for
  /// instance.)  This API is not meant for StackFrameLists that have unwinders
  /// and partake in lazy stack filling (using GetFramesUpTo).  Rather if you
  /// are building StackFrameLists with this API, you should build the entire
  /// list before making it available for use.
  bool SetFrameAtIndex(uint32_t idx, lldb::StackFrameSP &frame_sp);

  /// Ensures that frames up to (and including) `end_idx` are realized in the
  /// StackFrameList.  `end_idx` can be larger than the actual number of frames,
  /// in which case all the frames will be fetched.  Acquires the writer end of
  /// the list mutex.
  /// Returns true if the function was interrupted, false otherwise.
  /// Callers should first check (under the shared mutex) whether we need to
  /// fetch frames or not.
  bool GetFramesUpTo(uint32_t end_idx, InterruptionControl allow_interrupt);

  // This should be called with either the reader or writer end of the list
  // mutex held:
  bool GetAllFramesFetched() const {
    return m_concrete_frames_fetched == UINT32_MAX;
  }

  // This should be called with the writer end of the list mutex held.
  void SetAllFramesFetched() { m_concrete_frames_fetched = UINT32_MAX; }

  bool DecrementCurrentInlinedDepth();

  void ResetCurrentInlinedDepth();

  uint32_t GetCurrentInlinedDepth();

  void SetCurrentInlinedDepth(uint32_t new_depth);

  /// Calls into the stack frame recognizers and stop info to set the most
  /// relevant frame.  This can call out to arbitrary user code so it can't
  /// hold the StackFrameList mutex.
  void SelectMostRelevantFrame();

  typedef std::vector<lldb::StackFrameSP> collection;
  typedef collection::iterator iterator;
  typedef collection::const_iterator const_iterator;

  /// The thread this frame list describes.
  Thread &m_thread;

  /// The old stack frame list.
  // TODO: The old stack frame list is used to fill in missing frame info
  // heuristically when it's otherwise unavailable (say, because the unwinder
  // fails). We should have stronger checks to make sure that this is a valid
  // source of information.
  lldb::StackFrameListSP m_prev_frames_sp;

  /// A mutex for this frame list.  The only public API that requires the
  /// unique lock is Clear.  All other clients take the shared lock, though
  /// if we need more frames we may swap shared for unique to fulfill that
  /// requirement.
  mutable std::shared_mutex m_list_mutex;

  // Setting the inlined depth should be protected against other attempts to
  // change it, but since it doesn't mutate the list itself, we can limit the
  // critical regions it produces by having a separate mutex.
  mutable std::mutex m_inlined_depth_mutex;

  /// A cache of frames. This may need to be updated when the program counter
  /// changes.
  collection m_frames;

  /// The currently selected frame. An optional is used to record whether anyone
  /// has set the selected frame on this stack yet. We only let recognizers
  /// change the frame if this is the first time GetSelectedFrame is called.
  std::optional<uint32_t> m_selected_frame_idx;

  /// The number of concrete frames fetched while filling the frame list. This
  /// is only used when synthetic frames are enabled.
  uint32_t m_concrete_frames_fetched;

  /// The number of synthetic function activations (invisible frames) expanded
  /// from the concrete frame #0 activation.
  // TODO: Use an optional instead of UINT32_MAX to denote invalid values.
  uint32_t m_current_inlined_depth;

  /// The program counter value at the currently selected synthetic activation.
  /// This is only valid if m_current_inlined_depth is valid.
  // TODO: Use an optional instead of UINT32_MAX to denote invalid values.
  lldb::addr_t m_current_inlined_pc;

  /// Whether or not to show synthetic (inline) frames. Immutable.
  const bool m_show_inlined_frames;

private:
  uint32_t SetSelectedFrameNoLock(lldb_private::StackFrame *frame);
  lldb::StackFrameSP
  GetFrameAtIndexNoLock(uint32_t idx,
                        std::shared_lock<std::shared_mutex> &guard);

  /// These two Fetch frames APIs and SynthesizeTailCallFrames are called in
  /// GetFramesUpTo, they are the ones that actually add frames.  They must be
  /// called with the writer end of the list mutex held.

  /// Returns true if fetching frames was interrupted, false otherwise.
  bool FetchFramesUpTo(uint32_t end_idx, InterruptionControl allow_interrupt);
  /// Not currently interruptible so returns void.
  void FetchOnlyConcreteFramesUpTo(uint32_t end_idx);
  void SynthesizeTailCallFrames(StackFrame &next_frame);

  StackFrameList(const StackFrameList &) = delete;
  const StackFrameList &operator=(const StackFrameList &) = delete;
};

} // namespace lldb_private

#endif // LLDB_TARGET_STACKFRAMELIST_H
