//===-- lib/Parser/instrumented-parser.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/instrumented-parser.h"
#include "flang/Parser/message.h"
#include "flang/Parser/provenance.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

namespace Fortran::parser {

void ParsingLog::clear() { perPos_.clear(); }

// In the logs, just use the addresses of the message texts to sort the
// map keys.
bool operator<(const MessageFixedText &x, const MessageFixedText &y) {
  return x.text().begin() < y.text().begin();
}

bool ParsingLog::Fails(
    const char *at, const MessageFixedText &tag, ParseState &state) {
  std::size_t offset{reinterpret_cast<std::size_t>(at)};
  auto posIter{perPos_.find(offset)};
  if (posIter == perPos_.end()) {
    return false;
  }
  auto tagIter{posIter->second.perTag.find(tag)};
  if (tagIter == posIter->second.perTag.end()) {
    return false;
  }
  auto &entry{tagIter->second};
  if (entry.deferred && !state.deferMessages()) {
    return false; // don't fail fast, we want to generate messages
  }
  ++entry.count;
  if (!state.deferMessages()) {
    state.messages().Copy(entry.messages);
  }
  return !entry.pass;
}

void ParsingLog::Note(const char *at, const MessageFixedText &tag, bool pass,
    const ParseState &state) {
  std::size_t offset{reinterpret_cast<std::size_t>(at)};
  auto &entry{perPos_[offset].perTag[tag]};
  if (++entry.count == 1) {
    entry.pass = pass;
    entry.deferred = state.deferMessages();
    if (!entry.deferred) {
      entry.messages.Copy(state.messages());
    }
  } else {
    CHECK(entry.pass == pass);
    if (entry.deferred && !state.deferMessages()) {
      entry.deferred = false;
      entry.messages.Copy(state.messages());
    }
  }
}

void ParsingLog::Dump(
    llvm::raw_ostream &o, const AllCookedSources &allCooked) const {
  for (const auto &posLog : perPos_) {
    const char *at{reinterpret_cast<const char *>(posLog.first)};
    for (const auto &tagLog : posLog.second.perTag) {
      Message{at, tagLog.first}.Emit(o, allCooked, true);
      auto &entry{tagLog.second};
      o << "  " << (entry.pass ? "pass" : "fail") << " " << entry.count << '\n';
      entry.messages.Emit(o, allCooked);
    }
  }
}
} // namespace Fortran::parser
