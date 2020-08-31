//===-- include/flang/Parser/instrumented-parser.h --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_INSTRUMENTED_PARSER_H_
#define FORTRAN_PARSER_INSTRUMENTED_PARSER_H_

#include "parse-state.h"
#include "user-state.h"
#include "flang/Parser/message.h"
#include "flang/Parser/provenance.h"
#include <cstddef>
#include <map>

namespace llvm {
class raw_ostream;
}

namespace Fortran::parser {

class ParsingLog {
public:
  ParsingLog() {}

  void clear();

  bool Fails(const char *at, const MessageFixedText &tag, ParseState &);
  void Note(const char *at, const MessageFixedText &tag, bool pass,
      const ParseState &);
  void Dump(llvm::raw_ostream &, const AllCookedSources &) const;

private:
  struct LogForPosition {
    struct Entry {
      Entry() {}
      bool pass{true};
      int count{0};
      bool deferred{false};
      Messages messages;
    };
    std::map<MessageFixedText, Entry> perTag;
  };
  std::map<std::size_t, LogForPosition> perPos_;
};

template <typename PA> class InstrumentedParser {
public:
  using resultType = typename PA::resultType;
  constexpr InstrumentedParser(const InstrumentedParser &) = default;
  constexpr InstrumentedParser(const MessageFixedText &tag, const PA &parser)
      : tag_{tag}, parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (UserState * ustate{state.userState()}) {
      if (ParsingLog * log{ustate->log()}) {
        const char *at{state.GetLocation()};
        if (log->Fails(at, tag_, state)) {
          return std::nullopt;
        }
        Messages messages{std::move(state.messages())};
        std::optional<resultType> result{parser_.Parse(state)};
        log->Note(at, tag_, result.has_value(), state);
        state.messages().Restore(std::move(messages));
        return result;
      }
    }
    return parser_.Parse(state);
  }

private:
  const MessageFixedText tag_;
  const PA parser_;
};

template <typename PA>
inline constexpr auto instrumented(
    const MessageFixedText &tag, const PA &parser) {
  return InstrumentedParser{tag, parser};
}
} // namespace Fortran::parser
#endif // FORTRAN_PARSER_INSTRUMENTED_PARSER_H_
