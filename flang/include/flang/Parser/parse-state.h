//===-- include/flang/Parser/parse-state.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_PARSE_STATE_H_
#define FORTRAN_PARSER_PARSE_STATE_H_

// Defines the ParseState type used as the argument for every parser's
// Parse member or static function.  Tracks source provenance, context,
// accumulated messages, and an arbitrary UserState instance for parsing
// attempts.  Must be efficient to duplicate and assign for backtracking
// and recovery during parsing!

#include "user-state.h"
#include "flang/Common/Fortran-features.h"
#include "flang/Common/idioms.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/message.h"
#include "flang/Parser/provenance.h"
#include <cstddef>
#include <cstring>
#include <list>
#include <memory>
#include <optional>
#include <utility>

namespace Fortran::parser {

using common::LanguageFeature;

class ParseState {
public:
  ParseState(const CookedSource &cooked)
      : p_{cooked.AsCharBlock().begin()}, limit_{cooked.AsCharBlock().end()} {}
  ParseState(const ParseState &that)
      : p_{that.p_}, limit_{that.limit_}, context_{that.context_},
        userState_{that.userState_}, inFixedForm_{that.inFixedForm_},
        anyErrorRecovery_{that.anyErrorRecovery_},
        anyConformanceViolation_{that.anyConformanceViolation_},
        deferMessages_{that.deferMessages_},
        anyDeferredMessages_{that.anyDeferredMessages_},
        anyTokenMatched_{that.anyTokenMatched_} {}
  ParseState(ParseState &&that)
      : p_{that.p_}, limit_{that.limit_}, messages_{std::move(that.messages_)},
        context_{std::move(that.context_)}, userState_{that.userState_},
        inFixedForm_{that.inFixedForm_},
        anyErrorRecovery_{that.anyErrorRecovery_},
        anyConformanceViolation_{that.anyConformanceViolation_},
        deferMessages_{that.deferMessages_},
        anyDeferredMessages_{that.anyDeferredMessages_},
        anyTokenMatched_{that.anyTokenMatched_} {}
  ParseState &operator=(const ParseState &that) {
    p_ = that.p_, limit_ = that.limit_, context_ = that.context_;
    userState_ = that.userState_, inFixedForm_ = that.inFixedForm_;
    anyErrorRecovery_ = that.anyErrorRecovery_;
    anyConformanceViolation_ = that.anyConformanceViolation_;
    deferMessages_ = that.deferMessages_;
    anyDeferredMessages_ = that.anyDeferredMessages_;
    anyTokenMatched_ = that.anyTokenMatched_;
    return *this;
  }
  ParseState &operator=(ParseState &&that) {
    p_ = that.p_, limit_ = that.limit_, messages_ = std::move(that.messages_);
    context_ = std::move(that.context_);
    userState_ = that.userState_, inFixedForm_ = that.inFixedForm_;
    anyErrorRecovery_ = that.anyErrorRecovery_;
    anyConformanceViolation_ = that.anyConformanceViolation_;
    deferMessages_ = that.deferMessages_;
    anyDeferredMessages_ = that.anyDeferredMessages_;
    anyTokenMatched_ = that.anyTokenMatched_;
    return *this;
  }

  const Messages &messages() const { return messages_; }
  Messages &messages() { return messages_; }

  const Message::Reference &context() const { return context_; }
  Message::Reference &context() { return context_; }

  bool anyErrorRecovery() const { return anyErrorRecovery_; }
  void set_anyErrorRecovery() { anyErrorRecovery_ = true; }

  bool anyConformanceViolation() const { return anyConformanceViolation_; }
  void set_anyConformanceViolation() { anyConformanceViolation_ = true; }

  UserState *userState() const { return userState_; }
  ParseState &set_userState(UserState *u) {
    userState_ = u;
    return *this;
  }

  bool inFixedForm() const { return inFixedForm_; }
  ParseState &set_inFixedForm(bool yes = true) {
    inFixedForm_ = yes;
    return *this;
  }

  bool deferMessages() const { return deferMessages_; }
  ParseState &set_deferMessages(bool yes = true) {
    deferMessages_ = yes;
    return *this;
  }

  bool anyDeferredMessages() const { return anyDeferredMessages_; }
  ParseState &set_anyDeferredMessages(bool yes = true) {
    anyDeferredMessages_ = yes;
    return *this;
  }

  bool anyTokenMatched() const { return anyTokenMatched_; }
  ParseState &set_anyTokenMatched(bool yes = true) {
    anyTokenMatched_ = yes;
    return *this;
  }

  const char *GetLocation() const { return p_; }

  void PushContext(MessageFixedText text) {
    auto m{new Message{p_, text}}; // reference-counted
    m->SetContext(context_.get());
    context_ = Message::Reference{m};
  }

  void PopContext() {
    CHECK(context_);
    context_ = context_->attachment();
  }

  template <typename... A> void Say(CharBlock range, A &&...args) {
    if (deferMessages_) {
      anyDeferredMessages_ = true;
    } else {
      messages_.Say(range, std::forward<A>(args)...).SetContext(context_.get());
    }
  }
  template <typename... A> void Say(const MessageFixedText &text, A &&...args) {
    Say(p_, text, std::forward<A>(args)...);
  }
  template <typename... A>
  void Say(const MessageExpectedText &text, A &&...args) {
    Say(p_, text, std::forward<A>(args)...);
  }

  void Nonstandard(LanguageFeature lf, const MessageFixedText &msg) {
    Nonstandard(p_, lf, msg);
  }
  void Nonstandard(
      CharBlock range, LanguageFeature lf, const MessageFixedText &msg) {
    anyConformanceViolation_ = true;
    if (userState_ && userState_->features().ShouldWarn(lf)) {
      Say(range, msg);
    }
  }
  bool IsNonstandardOk(LanguageFeature lf, const MessageFixedText &msg) {
    if (userState_ && !userState_->features().IsEnabled(lf)) {
      return false;
    }
    Nonstandard(lf, msg);
    return true;
  }

  bool IsAtEnd() const { return p_ >= limit_; }

  const char *UncheckedAdvance(std::size_t n = 1) {
    const char *result{p_};
    p_ += n;
    return result;
  }

  std::optional<const char *> GetNextChar() {
    if (p_ < limit_) {
      return UncheckedAdvance();
    } else {
      return std::nullopt;
    }
  }

  std::optional<const char *> PeekAtNextChar() const {
    if (p_ < limit_) {
      return p_;
    } else {
      return std::nullopt;
    }
  }

  std::size_t BytesRemaining() const {
    std::size_t remain = limit_ - p_;
    return remain;
  }

  void CombineFailedParses(ParseState &&prev) {
    if (prev.anyTokenMatched_) {
      if (!anyTokenMatched_ || prev.p_ > p_) {
        anyTokenMatched_ = true;
        p_ = prev.p_;
        messages_ = std::move(prev.messages_);
      } else if (prev.p_ == p_) {
        messages_.Merge(std::move(prev.messages_));
      }
    }
    anyDeferredMessages_ |= prev.anyDeferredMessages_;
    anyConformanceViolation_ |= prev.anyConformanceViolation_;
    anyErrorRecovery_ |= prev.anyErrorRecovery_;
  }

private:
  // Text remaining to be parsed
  const char *p_{nullptr}, *limit_{nullptr};

  // Accumulated messages and current nested context.
  Messages messages_;
  Message::Reference context_;

  UserState *userState_{nullptr};

  bool inFixedForm_{false};
  bool anyErrorRecovery_{false};
  bool anyConformanceViolation_{false};
  bool deferMessages_{false};
  bool anyDeferredMessages_{false};
  bool anyTokenMatched_{false};
  // NOTE: Any additions or modifications to these data members must also be
  // reflected in the copy and move constructors defined at the top of this
  // class definition!
};
} // namespace Fortran::parser
#endif // FORTRAN_PARSER_PARSE_STATE_H_
