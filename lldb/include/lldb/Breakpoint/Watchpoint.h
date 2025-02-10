//===-- Watchpoint.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_BREAKPOINT_WATCHPOINT_H
#define LLDB_BREAKPOINT_WATCHPOINT_H

#include <memory>
#include <string>

#include "lldb/Breakpoint/StoppointSite.h"
#include "lldb/Breakpoint/WatchpointOptions.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/UserID.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class Watchpoint : public std::enable_shared_from_this<Watchpoint>,
                   public StoppointSite {
public:
  class WatchpointEventData : public EventData {
  public:
    WatchpointEventData(lldb::WatchpointEventType sub_type,
                        const lldb::WatchpointSP &new_watchpoint_sp);

    ~WatchpointEventData() override;

    static llvm::StringRef GetFlavorString();

    llvm::StringRef GetFlavor() const override;

    lldb::WatchpointEventType GetWatchpointEventType() const;

    lldb::WatchpointSP &GetWatchpoint();

    void Dump(Stream *s) const override;

    static lldb::WatchpointEventType
    GetWatchpointEventTypeFromEvent(const lldb::EventSP &event_sp);

    static lldb::WatchpointSP
    GetWatchpointFromEvent(const lldb::EventSP &event_sp);

    static const WatchpointEventData *
    GetEventDataFromEvent(const Event *event_sp);

  private:
    lldb::WatchpointEventType m_watchpoint_event;
    lldb::WatchpointSP m_new_watchpoint_sp;

    WatchpointEventData(const WatchpointEventData &) = delete;
    const WatchpointEventData &operator=(const WatchpointEventData &) = delete;
  };

  // Make sure watchpoint is properly disabled and subsequently enabled while
  // performing watchpoint actions.
  class WatchpointSentry {
  public:
    WatchpointSentry(lldb::ProcessSP p_sp, lldb::WatchpointSP w_sp)
        : process_sp(p_sp), watchpoint_sp(w_sp) {
      lldbassert(process_sp && watchpoint_sp && "Ill-formed WatchpointSentry!");

      constexpr bool notify = false;
      watchpoint_sp->TurnOnEphemeralMode();
      process_sp->DisableWatchpoint(watchpoint_sp, notify);
      process_sp->AddPreResumeAction(SentryPreResumeAction, this);
    }

    void DoReenable() {
      bool was_disabled = watchpoint_sp->IsDisabledDuringEphemeralMode();
      watchpoint_sp->TurnOffEphemeralMode();
      constexpr bool notify = false;
      if (was_disabled) {
        process_sp->DisableWatchpoint(watchpoint_sp, notify);
      } else {
        process_sp->EnableWatchpoint(watchpoint_sp, notify);
      }
    }

    ~WatchpointSentry() {
      DoReenable();
      process_sp->ClearPreResumeAction(SentryPreResumeAction, this);
    }

    static bool SentryPreResumeAction(void *sentry_void) {
      WatchpointSentry *sentry = static_cast<WatchpointSentry *>(sentry_void);
      sentry->DoReenable();
      return true;
    }

  private:
    lldb::ProcessSP process_sp;
    lldb::WatchpointSP watchpoint_sp;
  };

  Watchpoint(Target &target, lldb::addr_t addr, uint32_t size,
             const CompilerType *type, lldb::WatchpointMode mode);

  Watchpoint(Target &target, llvm::StringRef expr, uint32_t size,
             ExecutionContext &exe_ctx);

  ~Watchpoint() override;

  bool IsEnabled() const;

  // This doesn't really enable/disable the watchpoint.   It is currently
  // just for use in the Process plugin's {Enable,Disable}Watchpoint, which
  // should be used instead.
  void SetEnabled(bool enabled, bool notify = true);

  bool IsHardware() const override {
    return m_mode == lldb::eWatchpointModeHardware;
  }

  bool ShouldStop(StoppointCallbackContext *context) override;

  bool WatchpointRead() const { return m_watch_type & LLDB_WATCH_TYPE_READ; }
  bool WatchpointWrite() const { return m_watch_type & LLDB_WATCH_TYPE_WRITE; }
  bool WatchpointModify() const {
    return m_watch_type & LLDB_WATCH_TYPE_MODIFY;
  }

  uint32_t GetIgnoreCount() const;
  void SetIgnoreCount(uint32_t n);
  void SetWatchpointType(uint32_t type, bool notify = true);
  void SetDeclInfo(const std::string &str);
  std::string GetWatchSpec() const;
  void SetWatchSpec(const std::string &str);

  // This function determines whether we should report a watchpoint value
  // change. Specifically, it checks the watchpoint condition (if present),
  // ignore count and so on.
  //
  // \param[in] exe_ctx This should represent the current execution context
  // where execution stopped. It's used only for watchpoint condition
  // evaluation.
  //
  // \return Returns true if we should report a watchpoint hit.
  bool WatchedValueReportable(const ExecutionContext &exe_ctx);

  // Snapshot management interface.
  bool IsWatchVariable() const;
  void SetWatchVariable(bool val);

  /// \struct WatchpointVariableContext
  /// \brief Represents the context of a watchpoint variable.
  ///
  /// This struct encapsulates the information related to a watchpoint variable,
  /// including the watch ID and the execution context in which it is being
  /// used. This struct is passed as a Baton to the \b
  /// VariableWatchpointDisabler breakpoint callback.
  struct WatchpointVariableContext {
    /// \brief Constructor for WatchpointVariableContext.
    /// \param watch_id The ID of the watchpoint.
    /// \param exe_ctx The execution context associated with the watchpoint.
    WatchpointVariableContext(lldb::watch_id_t watch_id,
                              ExecutionContext exe_ctx)
        : watch_id(watch_id), exe_ctx(exe_ctx) {}

    lldb::watch_id_t watch_id; ///< The ID of the watchpoint.
    ExecutionContext
        exe_ctx; ///< The execution context associated with the watchpoint.
  };

  class WatchpointVariableBaton : public TypedBaton<WatchpointVariableContext> {
  public:
    WatchpointVariableBaton(std::unique_ptr<WatchpointVariableContext> Data)
        : TypedBaton(std::move(Data)) {}
  };

  bool SetupVariableWatchpointDisabler(lldb::StackFrameSP frame_sp) const;

  /// Callback routine to disable the watchpoint set on a local variable when
  ///  it goes out of scope.
  static bool VariableWatchpointDisabler(
      void *baton, lldb_private::StoppointCallbackContext *context,
      lldb::user_id_t break_id, lldb::user_id_t break_loc_id);

  void GetDescription(Stream *s, lldb::DescriptionLevel level) const;
  void Dump(Stream *s) const override;
  bool DumpSnapshots(Stream *s, const char *prefix = nullptr) const;
  void DumpWithLevel(Stream *s, lldb::DescriptionLevel description_level) const;
  Target &GetTarget() { return m_target; }
  const Status &GetError() { return m_error; }

  /// Returns the WatchpointOptions structure set for this watchpoint.
  ///
  /// \return
  ///     A pointer to this watchpoint's WatchpointOptions.
  WatchpointOptions *GetOptions() { return &m_options; }

  /// Set the callback action invoked when the watchpoint is hit.
  ///
  /// \param[in] callback
  ///    The method that will get called when the watchpoint is hit.
  /// \param[in] callback_baton
  ///    A void * pointer that will get passed back to the callback function.
  /// \param[in] is_synchronous
  ///    If \b true the callback will be run on the private event thread
  ///    before the stop event gets reported.  If false, the callback will get
  ///    handled on the public event thread after the stop has been posted.
  void SetCallback(WatchpointHitCallback callback, void *callback_baton,
                   bool is_synchronous = false);

  void SetCallback(WatchpointHitCallback callback,
                   const lldb::BatonSP &callback_baton_sp,
                   bool is_synchronous = false);

  void ClearCallback();

  /// Invoke the callback action when the watchpoint is hit.
  ///
  /// \param[in] context
  ///     Described the watchpoint event.
  ///
  /// \return
  ///     \b true if the target should stop at this watchpoint and \b false not.
  bool InvokeCallback(StoppointCallbackContext *context);

  // Condition
  /// Set the watchpoint's condition.
  ///
  /// \param[in] condition
  ///    The condition expression to evaluate when the watchpoint is hit.
  ///    Pass in nullptr to clear the condition.
  void SetCondition(const char *condition);

  /// Return a pointer to the text of the condition expression.
  ///
  /// \return
  ///    A pointer to the condition expression text, or nullptr if no
  //     condition has been set.
  const char *GetConditionText() const;

  void TurnOnEphemeralMode();

  void TurnOffEphemeralMode();

  bool IsDisabledDuringEphemeralMode();

  CompilerType GetCompilerType() const;

private:
  friend class Target;
  friend class WatchpointList;

  lldb::ValueObjectSP CalculateWatchedValue() const;

  void CaptureWatchedValue(lldb::ValueObjectSP new_value_sp) {
    m_old_value_sp = m_new_value_sp;
    m_new_value_sp = new_value_sp;
  }

  bool CheckWatchpointCondition(const ExecutionContext &exe_ctx) const;

  // This class facilitates retrieving a watchpoint's watched value.
  //
  // It's used by both hardware and software watchpoints to access
  // values stored in the process memory.
  //
  // To retrieve the value located in the memory, the value's memory address
  // and its CompilerType are required. ExecutionContext in this case should
  // contain information about current process, so CalculateWatchedValue
  // function first of all create ExecutionContext from the process of m_target.
  class AddressWatchpointCalculateStrategy {
  public:
    AddressWatchpointCalculateStrategy(Target &target, lldb::addr_t addr,
                                       uint32_t size, const CompilerType *type)
        : m_target{target}, m_addr{addr},
          m_type{CreateCompilerType(target, size, type)} {}

    lldb::ValueObjectSP CalculateWatchedValue() const;

    CompilerType GetCompilerType() const { return m_type; };

  private:
    static CompilerType CreateCompilerType(Target &target, uint32_t size,
                                           const CompilerType *type) {
      if (type && type->IsValid())
        return *type;
      // If we don't have a known type, then we force it to unsigned int of the
      // right size.
      return DeriveCompilerType(target, size);
    }

    static CompilerType DeriveCompilerType(Target &target, uint32_t size);

    Target &m_target;
    lldb::addr_t m_addr;
    CompilerType m_type;
  };

  // This class facilitates retrieving a watchpoint's watched value.
  //
  // It's used only by software watchpoints to obtain arbitral watched
  // value, in particular not stored in the process memory.
  //
  // To retrieve such values, this class evaluates watchpoint's exression,
  // therefor it is required that ExecutionContext should know about
  // stack frame in which watched expression was specified.
  class ExpressionWatchpointCalculateStrategy {
  public:
    ExpressionWatchpointCalculateStrategy(Target &target, llvm::StringRef expr,
                                          ExecutionContext exe_ctx)
        : m_target{target}, m_expr{expr}, m_exe_ctx{exe_ctx} {
      lldbassert(
          m_exe_ctx.GetFramePtr() &&
          "ExecutionContext should contain information about stack frame");
    }

    lldb::ValueObjectSP CalculateWatchedValue() const;

  private:
    Target &m_target;
    llvm::StringRef m_expr;     // ref on watchpoint's m_watch_spec_str
    ExecutionContext m_exe_ctx; // The execution context associated
                                // with watched value.
  };

  using WatchpointCalculateStrategy =
      std::variant<AddressWatchpointCalculateStrategy,
                   ExpressionWatchpointCalculateStrategy>;

  void ResetHistoricValues() {
    m_old_value_sp.reset();
    m_new_value_sp.reset();
  }

  Target &m_target;
  bool m_enabled;              // Is this watchpoint enabled
  lldb::WatchpointMode m_mode; // Is this hardware or software watchpoint
  bool m_is_watch_variable;    // True if set via 'watchpoint set variable'.
  bool m_is_ephemeral; // True if the watchpoint is in the ephemeral mode,
                       // meaning that it is
  // undergoing a pair of temporary disable/enable actions to avoid recursively
  // triggering further watchpoint events.
  uint32_t m_disabled_count; // Keep track of the count that the watchpoint is
                             // disabled while in ephemeral mode.
  // At the end of the ephemeral mode when the watchpoint is to be enabled
  // again, we check the count, if it is more than 1, it means the user-
  // supplied actions actually want the watchpoint to be disabled!
  uint32_t m_watch_type;
  uint32_t m_ignore_count;      // Number of times to ignore this watchpoint
  std::string m_watch_spec_str; // Spec for the watchpoint. It is optional for a
                                // hardware watchpoint, in which it is used only
                                // for dumping, but required for a software
                                // watchpoint calculation
  WatchpointCalculateStrategy m_calculate_strategy;
  std::string m_decl_str; // Declaration information, if any.
  lldb::ValueObjectSP m_old_value_sp;
  lldb::ValueObjectSP m_new_value_sp;
  lldb::ValueObjectSP
      m_previous_hit_value_sp; // Used in software watchpoints to ensure proper
                               // ignore count behavior
  Status m_error; // An error object describing errors associated with this
                  // watchpoint.
  WatchpointOptions m_options; // Settable watchpoint options, which is a
                               // delegate to handle the callback machinery.
  std::unique_ptr<UserExpression> m_condition_up; // The condition to test.

  void SetID(lldb::watch_id_t id) { m_id = id; }

  void SendWatchpointChangedEvent(lldb::WatchpointEventType eventKind);

  Watchpoint(const Watchpoint &) = delete;
  const Watchpoint &operator=(const Watchpoint &) = delete;
};

} // namespace lldb_private

#endif // LLDB_BREAKPOINT_WATCHPOINT_H
