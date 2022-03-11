//===-- lib/Parser/message.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/message.h"
#include "flang/Common/idioms.h"
#include "flang/Parser/char-set.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace Fortran::parser {

llvm::raw_ostream &operator<<(llvm::raw_ostream &o, const MessageFixedText &t) {
  std::size_t n{t.text().size()};
  for (std::size_t j{0}; j < n; ++j) {
    o << t.text()[j];
  }
  return o;
}

void MessageFormattedText::Format(const MessageFixedText *text, ...) {
  const char *p{text->text().begin()};
  std::string asString;
  if (*text->text().end() != '\0') {
    // not NUL-terminated
    asString = text->text().NULTerminatedToString();
    p = asString.c_str();
  }
  va_list ap;
  va_start(ap, text);
#ifdef _MSC_VER
  // Microsoft has a separate function for "positional arguments", which is
  // used in some messages.
  int need{_vsprintf_p(nullptr, 0, p, ap)};
#else
  int need{vsnprintf(nullptr, 0, p, ap)};
#endif

  CHECK(need >= 0);
  char *buffer{
      static_cast<char *>(std::malloc(static_cast<std::size_t>(need) + 1))};
  CHECK(buffer);
  va_end(ap);
  va_start(ap, text);
#ifdef _MSC_VER
  // Use positional argument variant of printf.
  int need2{_vsprintf_p(buffer, need + 1, p, ap)};
#else
  int need2{vsnprintf(buffer, need + 1, p, ap)};
#endif
  CHECK(need2 == need);
  va_end(ap);
  string_ = buffer;
  std::free(buffer);
  conversions_.clear();
}

const char *MessageFormattedText::Convert(const std::string &s) {
  conversions_.emplace_front(s);
  return conversions_.front().c_str();
}

const char *MessageFormattedText::Convert(std::string &s) {
  conversions_.emplace_front(s);
  return conversions_.front().c_str();
}

const char *MessageFormattedText::Convert(std::string &&s) {
  conversions_.emplace_front(std::move(s));
  return conversions_.front().c_str();
}

const char *MessageFormattedText::Convert(CharBlock x) {
  return Convert(x.ToString());
}

std::string MessageExpectedText::ToString() const {
  return std::visit(
      common::visitors{
          [](CharBlock cb) {
            return MessageFormattedText("expected '%s'"_err_en_US, cb)
                .MoveString();
          },
          [](const SetOfChars &set) {
            SetOfChars expect{set};
            if (expect.Has('\n')) {
              expect = expect.Difference('\n');
              if (expect.empty()) {
                return "expected end of line"_err_en_US.text().ToString();
              } else {
                std::string s{expect.ToString()};
                if (s.size() == 1) {
                  return MessageFormattedText(
                      "expected end of line or '%s'"_err_en_US, s)
                      .MoveString();
                } else {
                  return MessageFormattedText(
                      "expected end of line or one of '%s'"_err_en_US, s)
                      .MoveString();
                }
              }
            }
            std::string s{expect.ToString()};
            if (s.size() != 1) {
              return MessageFormattedText("expected one of '%s'"_err_en_US, s)
                  .MoveString();
            } else {
              return MessageFormattedText("expected '%s'"_err_en_US, s)
                  .MoveString();
            }
          },
      },
      u_);
}

bool MessageExpectedText::Merge(const MessageExpectedText &that) {
  return std::visit(common::visitors{
                        [](SetOfChars &s1, const SetOfChars &s2) {
                          s1 = s1.Union(s2);
                          return true;
                        },
                        [](const auto &, const auto &) { return false; },
                    },
      u_, that.u_);
}

bool Message::SortBefore(const Message &that) const {
  // Messages from prescanning have ProvenanceRange values for their locations,
  // while messages from later phases have CharBlock values, since the
  // conversion of cooked source stream locations to provenances is not
  // free and needs to be deferred, and many messages created during parsing
  // are speculative.  Messages with ProvenanceRange locations are ordered
  // before others for sorting.
  return std::visit(
      common::visitors{
          [](CharBlock cb1, CharBlock cb2) {
            return cb1.begin() < cb2.begin();
          },
          [](CharBlock, const ProvenanceRange &) { return false; },
          [](const ProvenanceRange &pr1, const ProvenanceRange &pr2) {
            return pr1.start() < pr2.start();
          },
          [](const ProvenanceRange &, CharBlock) { return true; },
      },
      location_, that.location_);
}

bool Message::IsFatal() const { return severity() == Severity::Error; }

Severity Message::severity() const {
  return std::visit(
      common::visitors{
          [](const MessageExpectedText &) { return Severity::Error; },
          [](const MessageFixedText &x) { return x.severity(); },
          [](const MessageFormattedText &x) { return x.severity(); },
      },
      text_);
}

std::string Message::ToString() const {
  return std::visit(
      common::visitors{
          [](const MessageFixedText &t) {
            return t.text().NULTerminatedToString();
          },
          [](const MessageFormattedText &t) { return t.string(); },
          [](const MessageExpectedText &e) { return e.ToString(); },
      },
      text_);
}

void Message::ResolveProvenances(const AllCookedSources &allCooked) {
  if (CharBlock * cb{std::get_if<CharBlock>(&location_)}) {
    if (std::optional<ProvenanceRange> resolved{
            allCooked.GetProvenanceRange(*cb)}) {
      location_ = *resolved;
    }
  }
  if (Message * attachment{attachment_.get()}) {
    attachment->ResolveProvenances(allCooked);
  }
}

std::optional<ProvenanceRange> Message::GetProvenanceRange(
    const AllCookedSources &allCooked) const {
  return std::visit(
      common::visitors{
          [&](CharBlock cb) { return allCooked.GetProvenanceRange(cb); },
          [](const ProvenanceRange &pr) { return std::make_optional(pr); },
      },
      location_);
}

void Message::Emit(llvm::raw_ostream &o, const AllCookedSources &allCooked,
    bool echoSourceLine) const {
  std::optional<ProvenanceRange> provenanceRange{GetProvenanceRange(allCooked)};
  std::string text;
  switch (severity()) {
  case Severity::Error:
    text = "error: ";
    break;
  case Severity::Warning:
    text = "warning: ";
    break;
  case Severity::Portability:
    text = "portability: ";
    break;
  case Severity::None:
    break;
  }
  text += ToString();
  const AllSources &sources{allCooked.allSources()};
  sources.EmitMessage(o, provenanceRange, text, echoSourceLine);
  bool isContext{attachmentIsContext_};
  for (const Message *attachment{attachment_.get()}; attachment;
       attachment = attachment->attachment_.get()) {
    text.clear();
    if (isContext) {
      text = "in the context: ";
    }
    text += attachment->ToString();
    sources.EmitMessage(
        o, attachment->GetProvenanceRange(allCooked), text, echoSourceLine);
    isContext = attachment->attachmentIsContext_;
  }
}

// Messages are equal if they're for the same location and text, and the user
// visible aspects of their attachments are the same
bool Message::operator==(const Message &that) const {
  if (!AtSameLocation(that) || ToString() != that.ToString()) {
    return false;
  }
  const Message *thatAttachment{that.attachment_.get()};
  for (const Message *attachment{attachment_.get()}; attachment;
       attachment = attachment->attachment_.get()) {
    if (!thatAttachment ||
        attachment->attachmentIsContext_ !=
            thatAttachment->attachmentIsContext_ ||
        *attachment != *thatAttachment) {
      return false;
    }
    thatAttachment = thatAttachment->attachment_.get();
  }
  return true;
}

bool Message::Merge(const Message &that) {
  return AtSameLocation(that) &&
      (!that.attachment_.get() ||
          attachment_.get() == that.attachment_.get()) &&
      std::visit(
          common::visitors{
              [](MessageExpectedText &e1, const MessageExpectedText &e2) {
                return e1.Merge(e2);
              },
              [](const auto &, const auto &) { return false; },
          },
          text_, that.text_);
}

Message &Message::Attach(Message *m) {
  if (!attachment_) {
    attachment_ = m;
  } else {
    if (attachment_->references() > 1) {
      // Don't attach to a shared context attachment; copy it first.
      attachment_ = new Message{*attachment_};
    }
    attachment_->Attach(m);
  }
  return *this;
}

Message &Message::Attach(std::unique_ptr<Message> &&m) {
  return Attach(m.release());
}

bool Message::AtSameLocation(const Message &that) const {
  return std::visit(
      common::visitors{
          [](CharBlock cb1, CharBlock cb2) {
            return cb1.begin() == cb2.begin();
          },
          [](const ProvenanceRange &pr1, const ProvenanceRange &pr2) {
            return pr1.start() == pr2.start();
          },
          [](const auto &, const auto &) { return false; },
      },
      location_, that.location_);
}

bool Messages::Merge(const Message &msg) {
  if (msg.IsMergeable()) {
    for (auto &m : messages_) {
      if (m.Merge(msg)) {
        return true;
      }
    }
  }
  return false;
}

void Messages::Merge(Messages &&that) {
  if (messages_.empty()) {
    *this = std::move(that);
  } else {
    while (!that.messages_.empty()) {
      if (Merge(that.messages_.front())) {
        that.messages_.pop_front();
      } else {
        auto next{that.messages_.begin()};
        ++next;
        messages_.splice(
            messages_.end(), that.messages_, that.messages_.begin(), next);
      }
    }
  }
}

void Messages::Copy(const Messages &that) {
  for (const Message &m : that.messages_) {
    Message copy{m};
    Say(std::move(copy));
  }
}

void Messages::ResolveProvenances(const AllCookedSources &allCooked) {
  for (Message &m : messages_) {
    m.ResolveProvenances(allCooked);
  }
}

void Messages::Emit(llvm::raw_ostream &o, const AllCookedSources &allCooked,
    bool echoSourceLines) const {
  std::vector<const Message *> sorted;
  for (const auto &msg : messages_) {
    sorted.push_back(&msg);
  }
  std::stable_sort(sorted.begin(), sorted.end(),
      [](const Message *x, const Message *y) { return x->SortBefore(*y); });
  const Message *lastMsg{nullptr};
  for (const Message *msg : sorted) {
    if (lastMsg && *msg == *lastMsg) {
      // Don't emit two identical messages for the same location
      continue;
    }
    msg->Emit(o, allCooked, echoSourceLines);
    lastMsg = msg;
  }
}

void Messages::AttachTo(Message &msg) {
  for (Message &m : messages_) {
    msg.Attach(std::move(m));
  }
  messages_.clear();
}

bool Messages::AnyFatalError() const {
  for (const auto &msg : messages_) {
    if (msg.IsFatal()) {
      return true;
    }
  }
  return false;
}
} // namespace Fortran::parser
