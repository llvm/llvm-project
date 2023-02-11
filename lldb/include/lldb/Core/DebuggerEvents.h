//===-- DebuggerEvents.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ModuleSpec.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Event.h"
#include "lldb/Utility/StructuredData.h"

#include <string>

#ifndef LLDB_CORE_DEBUGGER_EVENTS_H
#define LLDB_CORE_DEBUGGER_EVENTS_H

namespace lldb_private {
class Stream;

class ProgressEventData : public EventData {
public:
  ProgressEventData(uint64_t progress_id, const std::string &message,
                    uint64_t completed, uint64_t total, bool debugger_specific)
      : m_message(message), m_id(progress_id), m_completed(completed),
        m_total(total), m_debugger_specific(debugger_specific) {}

  static ConstString GetFlavorString();

  ConstString GetFlavor() const override;

  void Dump(Stream *s) const override;

  static const ProgressEventData *GetEventDataFromEvent(const Event *event_ptr);

  static StructuredData::DictionarySP
  GetAsStructuredData(const Event *event_ptr);

  uint64_t GetID() const { return m_id; }
  bool IsFinite() const { return m_total != UINT64_MAX; }
  uint64_t GetCompleted() const { return m_completed; }
  uint64_t GetTotal() const { return m_total; }
  const std::string &GetMessage() const { return m_message; }
  bool IsDebuggerSpecific() const { return m_debugger_specific; }

private:
  std::string m_message;
  const uint64_t m_id;
  uint64_t m_completed;
  const uint64_t m_total;
  const bool m_debugger_specific;
  ProgressEventData(const ProgressEventData &) = delete;
  const ProgressEventData &operator=(const ProgressEventData &) = delete;
};

class DiagnosticEventData : public EventData {
public:
  enum class Type {
    Info,
    Warning,
    Error,
  };
  DiagnosticEventData(Type type, std::string message, bool debugger_specific)
      : m_message(std::move(message)), m_type(type),
        m_debugger_specific(debugger_specific) {}
  ~DiagnosticEventData() override = default;

  const std::string &GetMessage() const { return m_message; }
  bool IsDebuggerSpecific() const { return m_debugger_specific; }
  Type GetType() const { return m_type; }

  llvm::StringRef GetPrefix() const;

  void Dump(Stream *s) const override;

  static ConstString GetFlavorString();
  ConstString GetFlavor() const override;

  static const DiagnosticEventData *
  GetEventDataFromEvent(const Event *event_ptr);

  static StructuredData::DictionarySP
  GetAsStructuredData(const Event *event_ptr);

protected:
  std::string m_message;
  Type m_type;
  const bool m_debugger_specific;

  DiagnosticEventData(const DiagnosticEventData &) = delete;
  const DiagnosticEventData &operator=(const DiagnosticEventData &) = delete;
};

class SymbolChangeEventData : public EventData {
public:
  SymbolChangeEventData(lldb::DebuggerWP debugger_wp, ModuleSpec module_spec)
      : m_debugger_wp(debugger_wp), m_module_spec(std::move(module_spec)) {}

  static ConstString GetFlavorString();
  ConstString GetFlavor() const override;

  static const SymbolChangeEventData *
  GetEventDataFromEvent(const Event *event_ptr);

  void DoOnRemoval(Event *event_ptr) override;

private:
  lldb::DebuggerWP m_debugger_wp;
  ModuleSpec m_module_spec;

  SymbolChangeEventData(const SymbolChangeEventData &) = delete;
  const SymbolChangeEventData &
  operator=(const SymbolChangeEventData &) = delete;
};

} // namespace lldb_private

#endif // LLDB_CORE_DEBUGGER_EVENTS_H
