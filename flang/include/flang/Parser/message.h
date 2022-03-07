//===-- include/flang/Parser/message.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_MESSAGE_H_
#define FORTRAN_PARSER_MESSAGE_H_

// Defines a representation for sequences of compiler messages.
// Supports nested contextualization.

#include "char-block.h"
#include "char-set.h"
#include "provenance.h"
#include "flang/Common/idioms.h"
#include "flang/Common/reference-counted.h"
#include "flang/Common/restorer.h"
#include <cstddef>
#include <cstring>
#include <forward_list>
#include <list>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace Fortran::parser {

// Use "..."_err_en_US, "..."_warn_en_US, and "..."_en_US literals to define
// the static text and fatality of a message.
enum class Severity { Error, Warning, Portability, None };

class MessageFixedText {
public:
  constexpr MessageFixedText() {}
  constexpr MessageFixedText(
      const char str[], std::size_t n, Severity severity = Severity::None)
      : text_{str, n}, severity_{severity} {}
  constexpr MessageFixedText(const MessageFixedText &) = default;
  constexpr MessageFixedText(MessageFixedText &&) = default;
  constexpr MessageFixedText &operator=(const MessageFixedText &) = default;
  constexpr MessageFixedText &operator=(MessageFixedText &&) = default;

  CharBlock text() const { return text_; }
  Severity severity() const { return severity_; }
  bool isFatal() const { return severity_ == Severity::Error; }

private:
  CharBlock text_;
  Severity severity_{Severity::None};
};

inline namespace literals {
constexpr MessageFixedText operator""_err_en_US(
    const char str[], std::size_t n) {
  return MessageFixedText{str, n, Severity::Error};
}
constexpr MessageFixedText operator""_warn_en_US(
    const char str[], std::size_t n) {
  return MessageFixedText{str, n, Severity::Warning};
}
constexpr MessageFixedText operator""_port_en_US(
    const char str[], std::size_t n) {
  return MessageFixedText{str, n, Severity::Portability};
}
constexpr MessageFixedText operator""_en_US(const char str[], std::size_t n) {
  return MessageFixedText{str, n, Severity::None};
}
} // namespace literals

// The construction of a MessageFormattedText uses a MessageFixedText
// as a vsnprintf() formatting string that is applied to the
// following arguments.  CharBlock and std::string argument
// values are also supported; they are automatically converted into
// char pointers that are suitable for '%s' formatting.
class MessageFormattedText {
public:
  template <typename... A>
  MessageFormattedText(const MessageFixedText &text, A &&...x)
      : severity_{text.severity()} {
    Format(&text, Convert(std::forward<A>(x))...);
  }
  MessageFormattedText(const MessageFormattedText &) = default;
  MessageFormattedText(MessageFormattedText &&) = default;
  MessageFormattedText &operator=(const MessageFormattedText &) = default;
  MessageFormattedText &operator=(MessageFormattedText &&) = default;
  const std::string &string() const { return string_; }
  bool isFatal() const { return severity_ == Severity::Error; }
  Severity severity() const { return severity_; }
  std::string MoveString() { return std::move(string_); }

private:
  void Format(const MessageFixedText *, ...);

  template <typename A> A Convert(const A &x) {
    static_assert(!std::is_class_v<std::decay_t<A>>);
    return x;
  }
  template <typename A> A Convert(A &x) {
    static_assert(!std::is_class_v<std::decay_t<A>>);
    return x;
  }
  template <typename A> common::IfNoLvalue<A, A> Convert(A &&x) {
    static_assert(!std::is_class_v<std::decay_t<A>>);
    return std::move(x);
  }
  const char *Convert(const char *s) { return s; }
  const char *Convert(char *s) { return s; }
  const char *Convert(const std::string &);
  const char *Convert(std::string &);
  const char *Convert(std::string &&);
  const char *Convert(CharBlock);
  std::intmax_t Convert(std::int64_t x) { return x; }
  std::uintmax_t Convert(std::uint64_t x) { return x; }

  Severity severity_;
  std::string string_;
  std::forward_list<std::string> conversions_; // preserves created strings
};

// Represents a formatted rendition of "expected '%s'"_err_en_US
// on a constant text or a set of characters.
class MessageExpectedText {
public:
  MessageExpectedText(const char *s, std::size_t n) {
    if (n == std::string::npos) {
      n = std::strlen(s);
    }
    if (n == 1) {
      // Treat a one-character string as a singleton set for better merging.
      u_ = SetOfChars{*s};
    } else {
      u_ = CharBlock{s, n};
    }
  }
  constexpr explicit MessageExpectedText(CharBlock cb) : u_{cb} {}
  constexpr explicit MessageExpectedText(char ch) : u_{SetOfChars{ch}} {}
  constexpr explicit MessageExpectedText(SetOfChars set) : u_{set} {}
  MessageExpectedText(const MessageExpectedText &) = default;
  MessageExpectedText(MessageExpectedText &&) = default;
  MessageExpectedText &operator=(const MessageExpectedText &) = default;
  MessageExpectedText &operator=(MessageExpectedText &&) = default;

  std::string ToString() const;
  bool Merge(const MessageExpectedText &);

private:
  std::variant<CharBlock, SetOfChars> u_;
};

class Message : public common::ReferenceCounted<Message> {
public:
  using Reference = common::CountedReference<Message>;

  Message(const Message &) = default;
  Message(Message &&) = default;
  Message &operator=(const Message &) = default;
  Message &operator=(Message &&) = default;

  Message(ProvenanceRange pr, const MessageFixedText &t)
      : location_{pr}, text_{t} {}
  Message(ProvenanceRange pr, const MessageFormattedText &s)
      : location_{pr}, text_{s} {}
  Message(ProvenanceRange pr, MessageFormattedText &&s)
      : location_{pr}, text_{std::move(s)} {}
  Message(ProvenanceRange pr, const MessageExpectedText &t)
      : location_{pr}, text_{t} {}

  Message(CharBlock csr, const MessageFixedText &t)
      : location_{csr}, text_{t} {}
  Message(CharBlock csr, const MessageFormattedText &s)
      : location_{csr}, text_{s} {}
  Message(CharBlock csr, MessageFormattedText &&s)
      : location_{csr}, text_{std::move(s)} {}
  Message(CharBlock csr, const MessageExpectedText &t)
      : location_{csr}, text_{t} {}

  template <typename RANGE, typename A, typename... As>
  Message(RANGE r, const MessageFixedText &t, A &&x, As &&...xs)
      : location_{r}, text_{MessageFormattedText{
                          t, std::forward<A>(x), std::forward<As>(xs)...}} {}

  bool attachmentIsContext() const { return attachmentIsContext_; }
  Reference attachment() const { return attachment_; }

  void SetContext(Message *c) {
    attachment_ = c;
    attachmentIsContext_ = true;
  }
  Message &Attach(Message *);
  Message &Attach(std::unique_ptr<Message> &&);
  template <typename... A> Message &Attach(A &&...args) {
    return Attach(new Message{std::forward<A>(args)...}); // reference-counted
  }

  bool SortBefore(const Message &that) const;
  bool IsFatal() const;
  Severity severity() const;
  std::string ToString() const;
  std::optional<ProvenanceRange> GetProvenanceRange(
      const AllCookedSources &) const;
  void Emit(llvm::raw_ostream &, const AllCookedSources &,
      bool echoSourceLine = true) const;

  // If this Message or any of its attachments locates itself via a CharBlock,
  // replace its location with the corresponding ProvenanceRange.
  void ResolveProvenances(const AllCookedSources &);

  bool IsMergeable() const {
    return std::holds_alternative<MessageExpectedText>(text_);
  }
  bool Merge(const Message &);
  bool operator==(const Message &that) const;
  bool operator!=(const Message &that) const { return !(*this == that); }

private:
  bool AtSameLocation(const Message &) const;
  std::variant<ProvenanceRange, CharBlock> location_;
  std::variant<MessageFixedText, MessageFormattedText, MessageExpectedText>
      text_;
  bool attachmentIsContext_{false};
  Reference attachment_;
};

class Messages {
public:
  Messages() {}
  Messages(Messages &&that) : messages_{std::move(that.messages_)} {}
  Messages &operator=(Messages &&that) {
    messages_ = std::move(that.messages_);
    return *this;
  }

  std::list<Message> &messages() { return messages_; }
  bool empty() const { return messages_.empty(); }
  void clear() { messages_.clear(); }

  template <typename... A> Message &Say(A &&...args) {
    return messages_.emplace_back(std::forward<A>(args)...);
  }

  void Annex(Messages &&that) {
    messages_.splice(messages_.end(), that.messages_);
  }

  bool Merge(const Message &);
  void Merge(Messages &&);
  void Copy(const Messages &);
  void ResolveProvenances(const AllCookedSources &);
  void Emit(llvm::raw_ostream &, const AllCookedSources &,
      bool echoSourceLines = true) const;
  void AttachTo(Message &);
  bool AnyFatalError() const;

private:
  std::list<Message> messages_;
};

class ContextualMessages {
public:
  ContextualMessages() = default;
  ContextualMessages(CharBlock at, Messages *m) : at_{at}, messages_{m} {}
  explicit ContextualMessages(Messages *m) : messages_{m} {}
  ContextualMessages(const ContextualMessages &that)
      : at_{that.at_}, messages_{that.messages_} {}

  CharBlock at() const { return at_; }
  Messages *messages() const { return messages_; }
  Message::Reference contextMessage() const { return contextMessage_; }
  bool empty() const { return !messages_ || messages_->empty(); }

  // Set CharBlock for messages; restore when the returned value is deleted
  common::Restorer<CharBlock> SetLocation(CharBlock at) {
    if (at.empty()) {
      at = at_;
    }
    return common::ScopedSet(at_, std::move(at));
  }

  common::Restorer<Message::Reference> SetContext(Message *m) {
    if (!m) {
      m = contextMessage_.get();
    }
    return common::ScopedSet(contextMessage_, m);
  }

  // Diverts messages to another buffer; restored when the returned
  // value is deleted.
  common::Restorer<Messages *> SetMessages(Messages &buffer) {
    return common::ScopedSet(messages_, &buffer);
  }
  // Discard future messages until the returned value is deleted.
  common::Restorer<Messages *> DiscardMessages() {
    return common::ScopedSet(messages_, nullptr);
  }

  template <typename... A> Message *Say(CharBlock at, A &&...args) {
    if (messages_ != nullptr) {
      auto &msg{messages_->Say(at, std::forward<A>(args)...)};
      if (contextMessage_) {
        msg.SetContext(contextMessage_.get());
      }
      return &msg;
    } else {
      return nullptr;
    }
  }

  template <typename... A>
  Message *Say(std::optional<CharBlock> at, A &&...args) {
    return Say(at.value_or(at_), std::forward<A>(args)...);
  }

  template <typename... A> Message *Say(A &&...args) {
    return Say(at_, std::forward<A>(args)...);
  }

private:
  CharBlock at_;
  Messages *messages_{nullptr};
  Message::Reference contextMessage_;
};
} // namespace Fortran::parser
#endif // FORTRAN_PARSER_MESSAGE_H_
