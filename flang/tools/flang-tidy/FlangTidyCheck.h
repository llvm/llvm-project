//===--- FlangTidyCheck.h - flang-tidy --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDYCHECK_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDYCHECK_H

#include "FlangTidyContext.h"
#include "FlangTidyOptions.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace Fortran::tidy {

/// This class should be specialized by any enum type that needs to be converted
/// to and from an \ref llvm::StringRef.
template <class T>
struct OptionEnumMapping {
  // Specializations of this struct must implement this function.
  static llvm::ArrayRef<std::pair<T, llvm::StringRef>>
  getEnumMapping() = delete;
};

/// This is the base class for all Flang Tidy checks. It provides a
/// common interface for all checks and allows them to interact with
/// the Flang Tidy context. Each check should inherit from this
/// class and implement the necessary methods to perform its
/// specific analysis.
///
/// For the user-facing documentation see:
/// https://flang.llvm.org/@PLACEHOLDER@/flang-tidy-checks.html
class FlangTidyCheck : public semantics::BaseChecker {
public:
  FlangTidyCheck(llvm::StringRef name, FlangTidyContext *context);
  virtual ~FlangTidyCheck() = default;
  llvm::StringRef name() const { return name_; }
  FlangTidyContext *context() { return context_; }
  bool fixAvailable() const { return fixAvailable_; }
  bool warningsAsErrors() const { return warningsAsErrors_; }

  using semantics::BaseChecker::Enter;
  using semantics::BaseChecker::Leave;

  /// Should store all options supported by this check with their
  /// current values or default values for options that haven't been overridden.
  ///
  /// The check should use ``Options.store()`` to store each option it supports
  /// whether it has the default value or it has been overridden.
  virtual void storeOptions(FlangTidyOptions::OptionMap &Options) {}

  /// Provides access to the ``FlangTidyCheck`` options via check-local
  /// names.
  ///
  /// Methods of this class prepend ``CheckName + "."`` to translate check-local
  /// option names to global option names.
  class OptionsView {
    void diagnoseBadIntegerOption(const llvm::Twine &Lookup,
                                  llvm::StringRef Unparsed) const;
    void diagnoseBadBooleanOption(const llvm::Twine &Lookup,
                                  llvm::StringRef Unparsed) const;
    void
    diagnoseBadEnumOption(const llvm::Twine &Lookup, llvm::StringRef Unparsed,
                          llvm::StringRef Suggestion = llvm::StringRef()) const;

  public:
    /// Initializes the instance using \p CheckName + "." as a prefix.
    OptionsView(llvm::StringRef CheckName,
                const FlangTidyOptions::OptionMap &CheckOptions,
                FlangTidyContext *Context);

    /// Read a named option from the ``Context``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is not present, return
    /// ``std::nullopt``.
    std::optional<llvm::StringRef> get(llvm::StringRef LocalName) const;

    /// Read a named option from the ``Context``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is not present, returns
    /// \p Default.
    llvm::StringRef get(llvm::StringRef LocalName,
                        llvm::StringRef Default) const;

    /// Read a named option from the ``Context``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either, return ``std::nullopt``.
    std::optional<llvm::StringRef>
    getLocalOrGlobal(llvm::StringRef LocalName) const;

    /// Read a named option from the ``Context``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either, returns \p Default.
    llvm::StringRef getLocalOrGlobal(llvm::StringRef LocalName,
                                     llvm::StringRef Default) const;

    /// Read a named option from the ``Context`` and parse it as an
    /// integral type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is not present,
    ///  return ``std::nullopt``.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return ``std::nullopt``.
    template <typename T>
    std::enable_if_t<std::is_integral_v<T>, std::optional<T>>
    get(llvm::StringRef LocalName) const {
      if (std::optional<llvm::StringRef> Value = get(LocalName)) {
        T Result{};
        if (!llvm::StringRef(*Value).getAsInteger(10, Result))
          return Result;
        diagnoseBadIntegerOption(NamePrefix + LocalName, *Value);
      }
      return std::nullopt;
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// integral type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is `none`, `null`,
    /// `-1` or empty, return ``std::nullopt``. If the corresponding
    /// key is not present, return \p Default.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return \p Default.
    template <typename T>
    std::enable_if_t<std::is_integral_v<T>, std::optional<T>>
    get(llvm::StringRef LocalName, std::optional<T> Default) const {
      if (std::optional<llvm::StringRef> Value = get(LocalName)) {
        if (Value == "" || Value == "none" || Value == "null" ||
            (std::is_unsigned_v<T> && Value == "-1"))
          return std::nullopt;
        T Result{};
        if (!llvm::StringRef(*Value).getAsInteger(10, Result))
          return Result;
        diagnoseBadIntegerOption(NamePrefix + LocalName, *Value);
      }
      return Default;
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// integral type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is not present, return
    /// \p Default.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return \p Default.
    template <typename T>
    std::enable_if_t<std::is_integral_v<T>, T> get(llvm::StringRef LocalName,
                                                   T Default) const {
      return get<T>(LocalName).value_or(Default);
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// integral type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either, return ``std::nullopt``.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return ``std::nullopt``.
    template <typename T>
    std::enable_if_t<std::is_integral_v<T>, std::optional<T>>
    getLocalOrGlobal(llvm::StringRef LocalName) const {
      std::optional<llvm::StringRef> ValueOr = get(LocalName);
      bool IsGlobal = false;
      if (!ValueOr) {
        IsGlobal = true;
        ValueOr = getLocalOrGlobal(LocalName);
        if (!ValueOr)
          return std::nullopt;
      }
      T Result{};
      if (!llvm::StringRef(*ValueOr).getAsInteger(10, Result))
        return Result;
      diagnoseBadIntegerOption(
          IsGlobal ? llvm::Twine(LocalName) : NamePrefix + LocalName, *ValueOr);
      return std::nullopt;
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// integral type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either, return \p Default. If the value value was found
    /// and equals ``none``, ``null``, ``-1`` or empty, return ``std::nullopt``.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return \p Default.
    template <typename T>
    std::enable_if_t<std::is_integral_v<T>, std::optional<T>>
    getLocalOrGlobal(llvm::StringRef LocalName,
                     std::optional<T> Default) const {
      std::optional<llvm::StringRef> ValueOr = get(LocalName);
      bool IsGlobal = false;
      if (!ValueOr) {
        IsGlobal = true;
        ValueOr = getLocalOrGlobal(LocalName);
        if (!ValueOr)
          return Default;
      }
      T Result{};
      if (ValueOr == "" || ValueOr == "none" || ValueOr == "null" ||
          (std::is_unsigned_v<T> && ValueOr == "-1"))
        return std::nullopt;
      if (!llvm::StringRef(*ValueOr).getAsInteger(10, Result))
        return Result;
      diagnoseBadIntegerOption(
          IsGlobal ? llvm::Twine(LocalName) : NamePrefix + LocalName, *ValueOr);
      return Default;
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// integral type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either, return \p Default.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return \p Default.
    template <typename T>
    std::enable_if_t<std::is_integral_v<T>, T>
    getLocalOrGlobal(llvm::StringRef LocalName, T Default) const {
      return getLocalOrGlobal<T>(LocalName).value_or(Default);
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// enum type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is not present, return
    /// ``std::nullopt``.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return ``std::nullopt``.
    ///
    /// \ref Fortran::tidy::OptionEnumMapping must be specialized for ``T`` to
    /// supply the mapping required to convert between ``T`` and a string.
    template <typename T>
    std::enable_if_t<std::is_enum_v<T>, std::optional<T>>
    get(llvm::StringRef LocalName) const {
      if (std::optional<int64_t> ValueOr =
              getEnumInt(LocalName, typeEraseMapping<T>(), false))
        return static_cast<T>(*ValueOr);
      return std::nullopt;
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// enum type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from the
    /// ``CheckOptions``. If the corresponding key is not present,
    /// return \p Default.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return \p Default.
    ///
    /// \ref Fortran::tidy::OptionEnumMapping must be specialized for ``T`` to
    /// supply the mapping required to convert between ``T`` and a string.
    template <typename T>
    std::enable_if_t<std::is_enum_v<T>, T> get(llvm::StringRef LocalName,
                                               T Default) const {
      return get<T>(LocalName).value_or(Default);
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// enum type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either, returns ``std::nullopt``.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return ``std::nullopt``.
    ///
    /// \ref Fortran::tidy::OptionEnumMapping must be specialized for ``T`` to
    /// supply the mapping required to convert between ``T`` and a string.
    template <typename T>
    std::enable_if_t<std::is_enum_v<T>, std::optional<T>>
    getLocalOrGlobal(llvm::StringRef LocalName) const {
      if (std::optional<int64_t> ValueOr =
              getEnumInt(LocalName, typeEraseMapping<T>(), true))
        return static_cast<T>(*ValueOr);
      return std::nullopt;
    }

    /// Read a named option from the ``Context`` and parse it as an
    /// enum type ``T``.
    ///
    /// Reads the option with the check-local name \p LocalName from local or
    /// global ``CheckOptions``. Gets local option first. If local is not
    /// present, falls back to get global option. If global option is not
    /// present either return \p Default.
    ///
    /// If the corresponding key can't be parsed as a ``T``, emit a
    /// diagnostic and return \p Default.
    ///
    /// \ref Fortran::tidy::OptionEnumMapping must be specialized for ``T`` to
    /// supply the mapping required to convert between ``T`` and a string.
    template <typename T>
    std::enable_if_t<std::is_enum_v<T>, T>
    getLocalOrGlobal(llvm::StringRef LocalName, T Default) const {
      return getLocalOrGlobal<T>(LocalName).value_or(Default);
    }

    /// Stores an option with the check-local name \p LocalName with
    /// string value \p Value to \p Options.
    void store(FlangTidyOptions::OptionMap &Options, llvm::StringRef LocalName,
               llvm::StringRef Value) const;

    /// Stores an option with the check-local name \p LocalName with
    /// integer value \p Value to \p Options.
    template <typename T>
    std::enable_if_t<std::is_integral_v<T>>
    store(FlangTidyOptions::OptionMap &Options, llvm::StringRef LocalName,
          T Value) const {
      if constexpr (std::is_signed_v<T>)
        storeInt(Options, LocalName, Value);
      else
        storeUnsigned(Options, LocalName, Value);
    }

    /// Stores an option with the check-local name \p LocalName with
    /// integer value \p Value to \p Options. If the value is empty
    /// stores "none"
    template <typename T>
    std::enable_if_t<std::is_integral_v<T>>
    store(FlangTidyOptions::OptionMap &Options, llvm::StringRef LocalName,
          std::optional<T> Value) const {
      if (Value)
        store(Options, LocalName, *Value);
      else
        store(Options, LocalName, "none");
    }

    /// Stores an option with the check-local name \p LocalName as the string
    /// representation of the Enum \p Value to \p Options.
    ///
    /// \ref Fortran::tidy::OptionEnumMapping must be specialized for ``T`` to
    /// supply the mapping required to convert between ``T`` and a string.
    template <typename T>
    std::enable_if_t<std::is_enum_v<T>>
    store(FlangTidyOptions::OptionMap &Options, llvm::StringRef LocalName,
          T Value) const {
      llvm::ArrayRef<std::pair<T, llvm::StringRef>> Mapping =
          OptionEnumMapping<T>::getEnumMapping();
      auto Iter = llvm::find_if(
          Mapping, [&](const std::pair<T, llvm::StringRef> &NameAndEnum) {
            return NameAndEnum.first == Value;
          });
      assert(Iter != Mapping.end() && "Unknown Case Value");
      store(Options, LocalName, Iter->second);
    }

  private:
    using NameAndValue = std::pair<int64_t, llvm::StringRef>;

    std::optional<int64_t> getEnumInt(llvm::StringRef LocalName,
                                      llvm::ArrayRef<NameAndValue> Mapping,
                                      bool CheckGlobal) const;

    template <typename T>
    std::enable_if_t<std::is_enum_v<T>, std::vector<NameAndValue>>
    typeEraseMapping() const {
      llvm::ArrayRef<std::pair<T, llvm::StringRef>> Mapping =
          OptionEnumMapping<T>::getEnumMapping();
      std::vector<NameAndValue> Result;
      Result.reserve(Mapping.size());
      for (auto &MappedItem : Mapping) {
        Result.emplace_back(static_cast<int64_t>(MappedItem.first),
                            MappedItem.second);
      }
      return Result;
    }

    void storeInt(FlangTidyOptions::OptionMap &Options,
                  llvm::StringRef LocalName, int64_t Value) const;

    void storeUnsigned(FlangTidyOptions::OptionMap &Options,
                       llvm::StringRef LocalName, uint64_t Value) const;

    std::string NamePrefix;
    const FlangTidyOptions::OptionMap &CheckOptions;
    FlangTidyContext *Context;
  };

  virtual void Enter(const parser::AcImpliedDo &) {}
  virtual void Enter(const parser::ArithmeticIfStmt &) {}
  virtual void Enter(const parser::AssignedGotoStmt &) {}
  virtual void Enter(const parser::AssignmentStmt &) {}
  virtual void Enter(const parser::AssignStmt &) {}
  virtual void Enter(const parser::AssociateConstruct &) {}
  virtual void Enter(const parser::BackspaceStmt &) {}
  virtual void Enter(const parser::Block &) {}
  virtual void Enter(const parser::CallStmt &) {}
  virtual void Enter(const parser::CaseConstruct &) {}
  virtual void Enter(const parser::CommonStmt &) {}
  virtual void Enter(const parser::ComputedGotoStmt &) {}
  virtual void Enter(const parser::DataImpliedDo &) {}
  virtual void Enter(const parser::DataStmt &) {}
  virtual void Enter(const parser::Designator &) {}
  virtual void Enter(const parser::DoConstruct &) {}
  virtual void Enter(const parser::EntityDecl &) {}
  virtual void Enter(const parser::ExecutableConstruct &) {}
  virtual void Enter(const parser::Expr &) {}
  virtual void Enter(const parser::Expr::AND &) {}
  virtual void Enter(const parser::Expr::OR &) {}
  virtual void Enter(const parser::Expr::Power &) {}
  virtual void Enter(const parser::ForallConstruct &) {}
  virtual void Enter(const parser::ForallStmt &) {}
  virtual void Enter(const parser::FunctionSubprogram &) {}
  virtual void Enter(const parser::GotoStmt &) {}
  virtual void Enter(const parser::IfConstruct &) {}
  virtual void Enter(const parser::IfStmt &) {}
  virtual void Enter(const parser::InputImpliedDo &) {}
  virtual void Enter(const parser::Name &) {}
  virtual void Enter(const parser::OmpAtomicUpdate &) {}
  virtual void Enter(const parser::OpenMPBlockConstruct &) {}
  virtual void Enter(const parser::OpenMPCriticalConstruct &) {}
  virtual void Enter(const parser::OpenMPLoopConstruct &) {}
  virtual void Enter(const parser::OutputImpliedDo &) {}
  virtual void Enter(const parser::PauseStmt &) {}
  virtual void Enter(const parser::SelectRankConstruct &) {}
  virtual void Enter(const parser::SelectTypeConstruct &) {}
  virtual void Enter(const parser::SubroutineSubprogram &) {}
  virtual void Enter(const parser::UseStmt &) {}
  virtual void Leave(const parser::AllocateStmt &) {}
  virtual void Leave(const parser::AssignmentStmt &) {}
  virtual void Leave(const parser::BackspaceStmt &) {}
  virtual void Leave(const parser::Block &) {}
  virtual void Leave(const parser::CloseStmt &) {}
  virtual void Leave(const parser::CommonStmt &) {}
  virtual void Leave(const parser::DeallocateStmt &) {}
  virtual void Leave(const parser::EndfileStmt &) {}
  virtual void Leave(const parser::EventPostStmt &) {}
  virtual void Leave(const parser::EventWaitStmt &) {}
  virtual void Leave(const parser::FileUnitNumber &) {}
  virtual void Leave(const parser::FlushStmt &) {}
  virtual void Leave(const parser::FormTeamStmt &) {}
  virtual void Leave(const parser::FunctionSubprogram &) {}
  virtual void Leave(const parser::InquireStmt &) {}
  virtual void Leave(const parser::LockStmt &) {}
  virtual void Leave(const parser::Name &) {}
  virtual void Leave(const parser::OmpAtomicUpdate &) {}
  virtual void Leave(const parser::OpenMPBlockConstruct &) {}
  virtual void Leave(const parser::OpenMPCriticalConstruct &) {}
  virtual void Leave(const parser::OpenMPLoopConstruct &) {}
  virtual void Leave(const parser::OpenStmt &) {}
  virtual void Leave(const parser::PointerAssignmentStmt &) {}
  virtual void Leave(const parser::PrintStmt &) {}
  virtual void Leave(const parser::Program &) {}
  virtual void Leave(const parser::ProgramUnit &) {}
  virtual void Leave(const parser::ReadStmt &) {}
  virtual void Leave(const parser::RewindStmt &) {}
  virtual void Leave(const parser::SubroutineSubprogram &) {}
  virtual void Leave(const parser::SyncAllStmt &) {}
  virtual void Leave(const parser::SyncImagesStmt &) {}
  virtual void Leave(const parser::SyncMemoryStmt &) {}
  virtual void Leave(const parser::SyncTeamStmt &) {}
  virtual void Leave(const parser::UnlockStmt &) {}
  virtual void Leave(const parser::UseStmt &) {}
  virtual void Leave(const parser::WaitStmt &) {}
  virtual void Leave(const parser::WriteStmt &) {}

  template <typename... Args>
  parser::Message &Say(parser::CharBlock at, parser::MessageFixedText &&message,
                       Args &&...args) {
    // construct a new fixedTextMessage
    std::string str{message.text().ToString()};
    str.append(" [%s]");
    parser::MessageFixedText newMessage{str.c_str(), str.length(),
                                        message.severity()};
    if (warningsAsErrors_) {
      newMessage.set_severity(parser::Severity::Error);
    }
    return context_->getSemanticsContext().Say(
        at, std::move(newMessage), std::forward<Args>(args)..., name_.str());
  }

protected:
  OptionsView Options;

private:
  bool fixAvailable_{false};
  llvm::StringRef name_;
  FlangTidyContext *context_;
  bool warningsAsErrors_;
};

/// Read a named option from the ``Context`` and parse it as a bool.
///
/// Reads the option with the check-local name \p LocalName from the
/// ``CheckOptions``. If the corresponding key is not present, return
/// ``std::nullopt``.
///
/// If the corresponding key can't be parsed as a bool, emit a
/// diagnostic and return ``std::nullopt``.
template <>
std::optional<bool>
FlangTidyCheck::OptionsView::get<bool>(llvm::StringRef LocalName) const;

/// Read a named option from the ``Context`` and parse it as a bool.
///
/// Reads the option with the check-local name \p LocalName from the
/// ``CheckOptions``. If the corresponding key is not present, return
/// \p Default.
///
/// If the corresponding key can't be parsed as a bool, emit a
/// diagnostic and return \p Default.
template <>
std::optional<bool> FlangTidyCheck::OptionsView::getLocalOrGlobal<bool>(
    llvm::StringRef LocalName) const;

/// Stores an option with the check-local name \p LocalName with
/// bool value \p Value to \p Options.
template <>
void FlangTidyCheck::OptionsView::store<bool>(
    FlangTidyOptions::OptionMap &Options, llvm::StringRef LocalName,
    bool Value) const;

} // namespace Fortran::tidy

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDYCHECK_H
