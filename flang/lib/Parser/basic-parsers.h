//===-- lib/Parser/basic-parsers.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_BASIC_PARSERS_H_
#define FORTRAN_PARSER_BASIC_PARSERS_H_

// Let a "parser" be an instance of any class that supports this
// type definition and member (or static) function:
//
//   using resultType = ...;
//   std::optional<resultType> Parse(ParseState &) const;
//
// which either returns a value to signify a successful recognition or else
// returns {} to signify failure.  On failure, the state cannot be assumed
// to still be valid, in general -- see below for exceptions.
//
// This header defines the fundamental parser class templates and helper
// template functions.  See parser-combinators.txt for documentation.

#include "flang/Common/Fortran-features.h"
#include "flang/Common/idioms.h"
#include "flang/Common/indirection.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-state.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/user-state.h"
#include <cstring>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace Fortran::parser {

// fail<A>("..."_err_en_US) returns a parser that never succeeds.  It reports an
// error message at the current position.  The result type is unused,
// but might have to be specified at the point of call to satisfy
// the type checker.  The state remains valid.
template <typename A> class FailParser {
public:
  using resultType = A;
  constexpr FailParser(const FailParser &) = default;
  constexpr explicit FailParser(MessageFixedText t) : text_{t} {}
  std::optional<A> Parse(ParseState &state) const {
    state.Say(text_);
    return std::nullopt;
  }

private:
  const MessageFixedText text_;
};

template <typename A = Success> inline constexpr auto fail(MessageFixedText t) {
  return FailParser<A>{t};
}

// pure(x) returns a parser that always succeeds, does not advance the
// parse, and returns a captured value x whose type must be copy-constructible.
//
// pure<A>() is essentially pure(A{}); it returns a default-constructed A{},
// and works even when A is not copy-constructible.
template <typename A> class PureParser {
public:
  using resultType = A;
  constexpr PureParser(const PureParser &) = default;
  constexpr explicit PureParser(A &&x) : value_(std::move(x)) {}
  std::optional<A> Parse(ParseState &) const { return value_; }

private:
  const A value_;
};

template <typename A> inline constexpr auto pure(A x) {
  return PureParser<A>(std::move(x));
}

template <typename A> class PureDefaultParser {
public:
  using resultType = A;
  constexpr PureDefaultParser(const PureDefaultParser &) = default;
  constexpr PureDefaultParser() {}
  std::optional<A> Parse(ParseState &) const { return std::make_optional<A>(); }
};

template <typename A> inline constexpr auto pure() {
  return PureDefaultParser<A>();
}

// If a is a parser, attempt(a) is the same parser, but on failure
// the ParseState is guaranteed to have been restored to its initial value.
template <typename A> class BacktrackingParser {
public:
  using resultType = typename A::resultType;
  constexpr BacktrackingParser(const BacktrackingParser &) = default;
  constexpr BacktrackingParser(const A &parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    Messages messages{std::move(state.messages())};
    ParseState backtrack{state};
    std::optional<resultType> result{parser_.Parse(state)};
    if (result) {
      state.messages().Annex(std::move(messages));
    } else {
      state = std::move(backtrack);
      state.messages() = std::move(messages);
    }
    return result;
  }

private:
  const A parser_;
};

template <typename A> inline constexpr auto attempt(const A &parser) {
  return BacktrackingParser<A>{parser};
}

// For any parser x, the parser returned by !x is one that succeeds when
// x fails, returning a useless (but present) result.  !x fails when x succeeds.
template <typename PA> class NegatedParser {
public:
  using resultType = Success;
  constexpr NegatedParser(const NegatedParser &) = default;
  constexpr NegatedParser(PA p) : parser_{p} {}
  std::optional<Success> Parse(ParseState &state) const {
    ParseState forked{state};
    forked.set_deferMessages(true);
    if (parser_.Parse(forked)) {
      return std::nullopt;
    }
    return Success{};
  }

private:
  const PA parser_;
};

template <typename PA, typename = typename PA::resultType>
constexpr auto operator!(PA p) {
  return NegatedParser<PA>(p);
}

// For any parser x, the parser returned by lookAhead(x) is one that succeeds
// or fails if x does, but the state is not modified.
template <typename PA> class LookAheadParser {
public:
  using resultType = Success;
  constexpr LookAheadParser(const LookAheadParser &) = default;
  constexpr LookAheadParser(PA p) : parser_{p} {}
  std::optional<Success> Parse(ParseState &state) const {
    ParseState forked{state};
    forked.set_deferMessages(true);
    if (parser_.Parse(forked)) {
      return Success{};
    }
    return std::nullopt;
  }

private:
  const PA parser_;
};

template <typename PA> inline constexpr auto lookAhead(PA p) {
  return LookAheadParser<PA>{p};
}

// If a is a parser, inContext("..."_en_US, a) runs it in a nested message
// context.
template <typename PA> class MessageContextParser {
public:
  using resultType = typename PA::resultType;
  constexpr MessageContextParser(const MessageContextParser &) = default;
  constexpr MessageContextParser(MessageFixedText t, PA p)
      : text_{t}, parser_{p} {}
  std::optional<resultType> Parse(ParseState &state) const {
    state.PushContext(text_);
    std::optional<resultType> result{parser_.Parse(state)};
    state.PopContext();
    return result;
  }

private:
  const MessageFixedText text_;
  const PA parser_;
};

template <typename PA>
inline constexpr auto inContext(MessageFixedText context, PA parser) {
  return MessageContextParser{context, parser};
}

// If a is a parser, withMessage("..."_en_US, a) runs it unchanged if it
// succeeds, and overrides its messages with a specific one if it fails and
// has matched no tokens.
template <typename PA> class WithMessageParser {
public:
  using resultType = typename PA::resultType;
  constexpr WithMessageParser(const WithMessageParser &) = default;
  constexpr WithMessageParser(MessageFixedText t, PA p)
      : text_{t}, parser_{p} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (state.deferMessages()) { // fast path
      std::optional<resultType> result{parser_.Parse(state)};
      if (!result) {
        state.set_anyDeferredMessages();
      }
      return result;
    }
    Messages messages{std::move(state.messages())};
    bool hadAnyTokenMatched{state.anyTokenMatched()};
    state.set_anyTokenMatched(false);
    std::optional<resultType> result{parser_.Parse(state)};
    bool emitMessage{false};
    if (result) {
      messages.Annex(std::move(state.messages()));
      if (hadAnyTokenMatched) {
        state.set_anyTokenMatched();
      }
    } else if (state.anyTokenMatched()) {
      emitMessage = state.messages().empty();
      messages.Annex(std::move(state.messages()));
    } else {
      emitMessage = true;
      if (hadAnyTokenMatched) {
        state.set_anyTokenMatched();
      }
    }
    state.messages() = std::move(messages);
    if (emitMessage) {
      state.Say(text_);
    }
    return result;
  }

private:
  const MessageFixedText text_;
  const PA parser_;
};

template <typename PA>
inline constexpr auto withMessage(MessageFixedText msg, PA parser) {
  return WithMessageParser{msg, parser};
}

// If a and b are parsers, then a >> b returns a parser that succeeds when
// b succeeds after a does so, but fails when either a or b does.  The
// result is taken from b.  Similarly, a / b also succeeds if both a and b
// do so, but the result is that returned by a.
template <typename PA, typename PB> class SequenceParser {
public:
  using resultType = typename PB::resultType;
  constexpr SequenceParser(const SequenceParser &) = default;
  constexpr SequenceParser(PA pa, PB pb) : pa_{pa}, pb2_{pb} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (pa_.Parse(state)) {
      return pb2_.Parse(state);
    } else {
      return std::nullopt;
    }
  }

private:
  const PA pa_;
  const PB pb2_;
};

template <typename PA, typename PB>
inline constexpr auto operator>>(PA pa, PB pb) {
  return SequenceParser<PA, PB>{pa, pb};
}

template <typename PA, typename PB> class FollowParser {
public:
  using resultType = typename PA::resultType;
  constexpr FollowParser(const FollowParser &) = default;
  constexpr FollowParser(PA pa, PB pb) : pa_{pa}, pb_{pb} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (std::optional<resultType> ax{pa_.Parse(state)}) {
      if (pb_.Parse(state)) {
        return ax;
      }
    }
    return std::nullopt;
  }

private:
  const PA pa_;
  const PB pb_;
};

template <typename PA, typename PB>
inline constexpr auto operator/(PA pa, PB pb) {
  return FollowParser<PA, PB>{pa, pb};
}

template <typename PA, typename... Ps> class AlternativesParser {
public:
  using resultType = typename PA::resultType;
  constexpr AlternativesParser(PA pa, Ps... ps) : ps_{pa, ps...} {}
  constexpr AlternativesParser(const AlternativesParser &) = default;
  std::optional<resultType> Parse(ParseState &state) const {
    Messages messages{std::move(state.messages())};
    ParseState backtrack{state};
    std::optional<resultType> result{std::get<0>(ps_).Parse(state)};
    if constexpr (sizeof...(Ps) > 0) {
      if (!result) {
        ParseRest<1>(result, state, backtrack);
      }
    }
    state.messages().Annex(std::move(messages));
    return result;
  }

private:
  template <int J>
  void ParseRest(std::optional<resultType> &result, ParseState &state,
      ParseState &backtrack) const {
    ParseState prevState{std::move(state)};
    state = backtrack;
    result = std::get<J>(ps_).Parse(state);
    if (!result) {
      state.CombineFailedParses(std::move(prevState));
      if constexpr (J < sizeof...(Ps)) {
        ParseRest<J + 1>(result, state, backtrack);
      }
    }
  }

  const std::tuple<PA, Ps...> ps_;
};

template <typename... Ps> inline constexpr auto first(Ps... ps) {
  return AlternativesParser<Ps...>{ps...};
}

template <typename PA, typename PB>
inline constexpr auto operator||(PA pa, PB pb) {
  return AlternativesParser<PA, PB>{pa, pb};
}

// If a and b are parsers, then recovery(a,b) returns a parser that succeeds if
// a does so, or if a fails and b succeeds.  If a succeeds, b is not attempted.
// All messages from the first parse are retained.
// The two parsers must return values of the same type.
template <typename PA, typename PB> class RecoveryParser {
public:
  using resultType = typename PA::resultType;
  static_assert(std::is_same_v<resultType, typename PB::resultType>);
  constexpr RecoveryParser(const RecoveryParser &) = default;
  constexpr RecoveryParser(PA pa, PB pb) : pa_{pa}, pb_{pb} {}
  std::optional<resultType> Parse(ParseState &state) const {
    bool originallyDeferred{state.deferMessages()};
    ParseState backtrack{state};
    if (!originallyDeferred && state.messages().empty() &&
        !state.anyErrorRecovery()) {
      // Fast path.  There are no messages or recovered errors in the incoming
      // state.  Attempt to parse with messages deferred, expecting that the
      // parse will succeed silently.
      state.set_deferMessages(true);
      if (std::optional<resultType> ax{pa_.Parse(state)}) {
        if (!state.anyDeferredMessages() && !state.anyErrorRecovery()) {
          state.set_deferMessages(false);
          return ax;
        }
      }
      state = backtrack;
    }
    Messages messages{std::move(state.messages())};
    if (std::optional<resultType> ax{pa_.Parse(state)}) {
      state.messages().Annex(std::move(messages));
      return ax;
    }
    messages.Annex(std::move(state.messages()));
    bool hadDeferredMessages{state.anyDeferredMessages()};
    bool anyTokenMatched{state.anyTokenMatched()};
    state = std::move(backtrack);
    state.set_deferMessages(true);
    std::optional<resultType> bx{pb_.Parse(state)};
    state.messages() = std::move(messages);
    state.set_deferMessages(originallyDeferred);
    if (anyTokenMatched) {
      state.set_anyTokenMatched();
    }
    if (hadDeferredMessages) {
      state.set_anyDeferredMessages();
    }
    if (bx) {
      // Error recovery situations must also produce messages.
      CHECK(state.anyDeferredMessages() || state.messages().AnyFatalError());
      state.set_anyErrorRecovery();
    }
    return bx;
  }

private:
  const PA pa_;
  const PB pb_;
};

template <typename PA, typename PB>
inline constexpr auto recovery(PA pa, PB pb) {
  return RecoveryParser<PA, PB>{pa, pb};
}

// If x is a parser, then many(x) returns a parser that always succeeds
// and whose value is a list, possibly empty, of the values returned from
// repeated application of x until it fails or does not advance the parse.
template <typename PA> class ManyParser {
  using paType = typename PA::resultType;

public:
  using resultType = std::list<paType>;
  constexpr ManyParser(const ManyParser &) = default;
  constexpr ManyParser(PA parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    resultType result;
    auto at{state.GetLocation()};
    while (std::optional<paType> x{parser_.Parse(state)}) {
      result.emplace_back(std::move(*x));
      if (state.GetLocation() <= at) {
        break; // no forward progress, don't loop
      }
      at = state.GetLocation();
    }
    return {std::move(result)};
  }

private:
  const BacktrackingParser<PA> parser_;
};

template <typename PA> inline constexpr auto many(PA parser) {
  return ManyParser<PA>{parser};
}

// If x is a parser, then some(x) returns a parser that succeeds if x does
// and whose value is a nonempty list of the values returned from repeated
// application of x until it fails or does not advance the parse.  In other
// words, some(x) is a variant of many(x) that has to succeed at least once.
template <typename PA> class SomeParser {
  using paType = typename PA::resultType;

public:
  using resultType = std::list<paType>;
  constexpr SomeParser(const SomeParser &) = default;
  constexpr SomeParser(PA parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    auto start{state.GetLocation()};
    if (std::optional<paType> first{parser_.Parse(state)}) {
      resultType result;
      result.emplace_back(std::move(*first));
      if (state.GetLocation() > start) {
        result.splice(result.end(), many(parser_).Parse(state).value());
      }
      return {std::move(result)};
    }
    return std::nullopt;
  }

private:
  const PA parser_;
};

template <typename PA> inline constexpr auto some(PA parser) {
  return SomeParser<PA>{parser};
}

// If x is a parser, skipMany(x) is equivalent to many(x) but with no result.
template <typename PA> class SkipManyParser {
public:
  using resultType = Success;
  constexpr SkipManyParser(const SkipManyParser &) = default;
  constexpr SkipManyParser(PA parser) : parser_{parser} {}
  std::optional<Success> Parse(ParseState &state) const {
    for (auto at{state.GetLocation()};
         parser_.Parse(state) && state.GetLocation() > at;
         at = state.GetLocation()) {
    }
    return Success{};
  }

private:
  const BacktrackingParser<PA> parser_;
};

template <typename PA> inline constexpr auto skipMany(PA parser) {
  return SkipManyParser<PA>{parser};
}

// If x is a parser, skipManyFast(x) is equivalent to skipMany(x).
// The parser x must always advance on success and never invalidate the
// state on failure.
template <typename PA> class SkipManyFastParser {
public:
  using resultType = Success;
  constexpr SkipManyFastParser(const SkipManyFastParser &) = default;
  constexpr SkipManyFastParser(PA parser) : parser_{parser} {}
  std::optional<Success> Parse(ParseState &state) const {
    while (parser_.Parse(state)) {
    }
    return Success{};
  }

private:
  const PA parser_;
};

template <typename PA> inline constexpr auto skipManyFast(PA parser) {
  return SkipManyFastParser<PA>{parser};
}

// If x is a parser returning some type A, then maybe(x) returns a
// parser that returns std::optional<A>, always succeeding.
template <typename PA> class MaybeParser {
  using paType = typename PA::resultType;

public:
  using resultType = std::optional<paType>;
  constexpr MaybeParser(const MaybeParser &) = default;
  constexpr MaybeParser(PA parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (resultType result{parser_.Parse(state)}) {
      // permit optional<optional<...>>
      return {std::move(result)};
    }
    return resultType{};
  }

private:
  const BacktrackingParser<PA> parser_;
};

template <typename PA> inline constexpr auto maybe(PA parser) {
  return MaybeParser<PA>{parser};
}

// If x is a parser, then defaulted(x) returns a parser that always
// succeeds.  When x succeeds, its result is that of x; otherwise, its
// result is a default-constructed value of x's result type.
template <typename PA> class DefaultedParser {
public:
  using resultType = typename PA::resultType;
  constexpr DefaultedParser(const DefaultedParser &) = default;
  constexpr DefaultedParser(PA p) : parser_{p} {}
  std::optional<resultType> Parse(ParseState &state) const {
    std::optional<std::optional<resultType>> ax{maybe(parser_).Parse(state)};
    if (ax.value()) { // maybe() always succeeds
      return std::move(*ax);
    }
    return resultType{};
  }

private:
  const BacktrackingParser<PA> parser_;
};

template <typename PA> inline constexpr auto defaulted(PA p) {
  return DefaultedParser<PA>(p);
}

// If a is a parser, and f is a function mapping an rvalue of a's result type
// to some other type T, then applyFunction(f, a) returns a parser that succeeds
// iff a does, and whose result value ax has been passed through the function;
// the final result is that returned by the call f(std::move(ax)).
//
// Function application is generalized to functions with more than one
// argument with applyFunction(f, a, b, ...) succeeding if all of the parsers
// a, b, &c. do so, and the result is the value of applying f to their
// results.
//
// applyLambda(f, ...) is the same concept extended to std::function<> functors.
// It is not constexpr.
//
// Member function application is supported by applyMem(&C::f, a).  If the
// parser a succeeds and returns some value ax of type C, the result is that
// returned by ax.f().  Additional parser arguments can be specified to supply
// their results to the member function call, so applyMem(&C::f, a, b) succeeds
// if both a and b do so and returns the result of calling ax.f(std::move(bx)).

// Runs a sequence of parsers until one fails or all have succeeded.
// Collects their results in a std::tuple<std::optional<>...>.
template <typename... PARSER>
using ApplyArgs = std::tuple<std::optional<typename PARSER::resultType>...>;

template <typename... PARSER, std::size_t... J>
inline bool ApplyHelperArgs(const std::tuple<PARSER...> &parsers,
    ApplyArgs<PARSER...> &args, ParseState &state, std::index_sequence<J...>) {
  return (... &&
      (std::get<J>(args) = std::get<J>(parsers).Parse(state),
          std::get<J>(args).has_value()));
}

// Applies a function to the arguments collected by ApplyHelperArgs.
template <typename RESULT, typename... PARSER>
using ApplicableFunctionPointer = RESULT (*)(typename PARSER::resultType &&...);
template <typename RESULT, typename... PARSER>
using ApplicableFunctionObject =
    const std::function<RESULT(typename PARSER::resultType &&...)> &;

template <template <typename...> class FUNCTION, typename RESULT,
    typename... PARSER, std::size_t... J>
inline RESULT ApplyHelperFunction(FUNCTION<RESULT, PARSER...> f,
    ApplyArgs<PARSER...> &&args, std::index_sequence<J...>) {
  return f(std::move(*std::get<J>(args))...);
}

template <template <typename...> class FUNCTION, typename RESULT,
    typename... PARSER>
class ApplyFunction {
  using funcType = FUNCTION<RESULT, PARSER...>;

public:
  using resultType = RESULT;
  constexpr ApplyFunction(const ApplyFunction &) = default;
  constexpr ApplyFunction(funcType f, PARSER... p)
      : function_{f}, parsers_{p...} {}
  std::optional<resultType> Parse(ParseState &state) const {
    ApplyArgs<PARSER...> results;
    using Sequence = std::index_sequence_for<PARSER...>;
    if (ApplyHelperArgs(parsers_, results, state, Sequence{})) {
      return ApplyHelperFunction<FUNCTION, RESULT, PARSER...>(
          function_, std::move(results), Sequence{});
    } else {
      return std::nullopt;
    }
  }

private:
  const funcType function_;
  const std::tuple<PARSER...> parsers_;
};

template <typename RESULT, typename... PARSER>
inline constexpr auto applyFunction(
    ApplicableFunctionPointer<RESULT, PARSER...> f, const PARSER &...parser) {
  return ApplyFunction<ApplicableFunctionPointer, RESULT, PARSER...>{
      f, parser...};
}

template <typename RESULT, typename... PARSER>
inline /* not constexpr */ auto applyLambda(
    ApplicableFunctionObject<RESULT, PARSER...> f, const PARSER &...parser) {
  return ApplyFunction<ApplicableFunctionObject, RESULT, PARSER...>{
      f, parser...};
}

// Member function application
template <typename MEMFUNC, typename OBJPARSER, typename... PARSER,
    std::size_t... J>
inline auto ApplyHelperMember(MEMFUNC mfp,
    ApplyArgs<OBJPARSER, PARSER...> &&args, std::index_sequence<J...>) {
  return ((*std::get<0>(args)).*mfp)(std::move(*std::get<J + 1>(args))...);
}

template <typename MEMFUNC, typename OBJPARSER, typename... PARSER>
class ApplyMemberFunction {
  static_assert(std::is_member_function_pointer_v<MEMFUNC>);
  using funcType = MEMFUNC;

public:
  using resultType =
      std::invoke_result_t<MEMFUNC, typename OBJPARSER::resultType, PARSER...>;

  constexpr ApplyMemberFunction(const ApplyMemberFunction &) = default;
  constexpr ApplyMemberFunction(MEMFUNC f, OBJPARSER o, PARSER... p)
      : function_{f}, parsers_{o, p...} {}
  std::optional<resultType> Parse(ParseState &state) const {
    ApplyArgs<OBJPARSER, PARSER...> results;
    using Sequence1 = std::index_sequence_for<OBJPARSER, PARSER...>;
    using Sequence2 = std::index_sequence_for<PARSER...>;
    if (ApplyHelperArgs(parsers_, results, state, Sequence1{})) {
      return ApplyHelperMember<MEMFUNC, OBJPARSER, PARSER...>(
          function_, std::move(results), Sequence2{});
    } else {
      return std::nullopt;
    }
  }

private:
  const funcType function_;
  const std::tuple<OBJPARSER, PARSER...> parsers_;
};

template <typename MEMFUNC, typename OBJPARSER, typename... PARSER>
inline constexpr auto applyMem(
    MEMFUNC memfn, const OBJPARSER &objParser, PARSER... parser) {
  return ApplyMemberFunction<MEMFUNC, OBJPARSER, PARSER...>{
      memfn, objParser, parser...};
}

// As is done with function application via applyFunction() above, class
// instance construction can also be based upon the results of successful
// parses.  For some type T and zero or more parsers a, b, &c., the call
// construct<T>(a, b, ...) returns a parser that succeeds if all of
// its argument parsers do so in succession, and whose result is an
// instance of T constructed upon the values they returned.
// With a single argument that is a parser with no usable value,
// construct<T>(p) invokes T's default nullary constructor (T(){}).
// (This means that "construct<T>(Foo >> Bar >> ok)" is functionally
// equivalent to "Foo >> Bar >> construct<T>()", but I'd like to hold open
// the opportunity to make construct<> capture source provenance all of the
// time, and the first form will then lead to better error positioning.)

template <typename RESULT, typename... PARSER, std::size_t... J>
inline RESULT ApplyHelperConstructor(
    ApplyArgs<PARSER...> &&args, std::index_sequence<J...>) {
  return RESULT{std::move(*std::get<J>(args))...};
}

template <typename RESULT, typename... PARSER> class ApplyConstructor {
public:
  using resultType = RESULT;
  constexpr ApplyConstructor(const ApplyConstructor &) = default;
  constexpr explicit ApplyConstructor(PARSER... p) : parsers_{p...} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if constexpr (sizeof...(PARSER) == 0) {
      return RESULT{};
    } else {
      if constexpr (sizeof...(PARSER) == 1) {
        return ParseOne(state);
      } else {
        ApplyArgs<PARSER...> results;
        using Sequence = std::index_sequence_for<PARSER...>;
        if (ApplyHelperArgs(parsers_, results, state, Sequence{})) {
          return ApplyHelperConstructor<RESULT, PARSER...>(
              std::move(results), Sequence{});
        }
      }
      return std::nullopt;
    }
  }

private:
  std::optional<resultType> ParseOne(ParseState &state) const {
    if constexpr (std::is_same_v<Success, typename PARSER::resultType...>) {
      if (std::get<0>(parsers_).Parse(state)) {
        return RESULT{};
      }
    } else if (auto arg{std::get<0>(parsers_).Parse(state)}) {
      return RESULT{std::move(*arg)};
    }
    return std::nullopt;
  }

  const std::tuple<PARSER...> parsers_;
};

template <typename RESULT, typename... PARSER>
inline constexpr auto construct(PARSER... p) {
  return ApplyConstructor<RESULT, PARSER...>{p...};
}

// For a parser p, indirect(p) returns a parser that builds an indirect
// reference to p's return type.
template <typename PA> inline constexpr auto indirect(PA p) {
  return construct<common::Indirection<typename PA::resultType>>(p);
}

// If a and b are parsers, then nonemptySeparated(a, b) returns a parser
// that succeeds if a does.  If a succeeds, it then applies many(b >> a).
// The result is the list of the values returned from all of the applications
// of a.
template <typename T>
common::IfNoLvalue<std::list<T>, T> prepend(T &&head, std::list<T> &&rest) {
  rest.push_front(std::move(head));
  return std::move(rest);
}

template <typename PA, typename PB> class NonemptySeparated {
private:
  using paType = typename PA::resultType;

public:
  using resultType = std::list<paType>;
  constexpr NonemptySeparated(const NonemptySeparated &) = default;
  constexpr NonemptySeparated(PA p, PB sep) : parser_{p}, separator_{sep} {}
  std::optional<resultType> Parse(ParseState &state) const {
    return applyFunction<std::list<paType>>(
        prepend<paType>, parser_, many(separator_ >> parser_))
        .Parse(state);
  }

private:
  const PA parser_;
  const PB separator_;
};

template <typename PA, typename PB>
inline constexpr auto nonemptySeparated(PA p, PB sep) {
  return NonemptySeparated<PA, PB>{p, sep};
}

// ok is a parser that always succeeds.  It is useful when a parser
// must discard its result in order to be compatible in type with other
// parsers in an alternative, e.g. "x >> ok || y >> ok" is type-safe even
// when x and y have distinct result types.
struct OkParser {
  using resultType = Success;
  constexpr OkParser() {}
  static constexpr std::optional<Success> Parse(ParseState &) {
    return Success{};
  }
};
constexpr OkParser ok;

// A variant of recovery() above for convenience.
template <typename PA, typename PB>
inline constexpr auto localRecovery(MessageFixedText msg, PA pa, PB pb) {
  return recovery(withMessage(msg, pa), pb >> pure<typename PA::resultType>());
}

// nextCh is a parser that succeeds if the parsing state is not
// at the end of its input, returning the next character location and
// advancing the parse when it does so.
struct NextCh {
  using resultType = const char *;
  constexpr NextCh() {}
  std::optional<const char *> Parse(ParseState &state) const {
    if (std::optional<const char *> result{state.GetNextChar()}) {
      return result;
    }
    state.Say("end of file"_err_en_US);
    return std::nullopt;
  }
};

constexpr NextCh nextCh;

// If a is a parser for some nonstandard language feature LF, extension<LF>(a)
// is a parser that optionally enabled, sets a strict conformance violation
// flag, and may emit a warning message, if those are enabled.
template <LanguageFeature LF, typename PA> class NonstandardParser {
public:
  using resultType = typename PA::resultType;
  constexpr NonstandardParser(const NonstandardParser &) = default;
  constexpr NonstandardParser(PA parser, MessageFixedText msg)
      : parser_{parser}, message_{msg} {}
  constexpr NonstandardParser(PA parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (UserState * ustate{state.userState()}) {
      if (!ustate->features().IsEnabled(LF)) {
        return std::nullopt;
      }
    }
    auto at{state.GetLocation()};
    auto result{parser_.Parse(state)};
    if (result && !message_.empty()) {
      state.Nonstandard(
          CharBlock{at, std::max(state.GetLocation(), at + 1)}, LF, message_);
    }
    return result;
  }

private:
  const PA parser_;
  const MessageFixedText message_;
};

template <LanguageFeature LF, typename PA>
inline constexpr auto extension(MessageFixedText feature, PA parser) {
  return NonstandardParser<LF, PA>(parser, feature);
}

template <LanguageFeature LF, typename PA>
inline constexpr auto extension(PA parser) {
  return NonstandardParser<LF, PA>(parser);
}

// If a is a parser for some deprecated or deleted language feature LF,
// deprecated<LF>(a) is a parser that is optionally enabled, sets a strict
// conformance violation flag, and may emit a warning message, if enabled.
template <LanguageFeature LF, typename PA> class DeprecatedParser {
public:
  using resultType = typename PA::resultType;
  constexpr DeprecatedParser(const DeprecatedParser &) = default;
  constexpr DeprecatedParser(PA parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (UserState * ustate{state.userState()}) {
      if (!ustate->features().IsEnabled(LF)) {
        return std::nullopt;
      }
    }
    auto at{state.GetLocation()};
    auto result{parser_.Parse(state)};
    if (result) {
      state.Nonstandard(CharBlock{at, state.GetLocation()}, LF,
          "deprecated usage"_port_en_US);
    }
    return result;
  }

private:
  const PA parser_;
};

template <LanguageFeature LF, typename PA>
inline constexpr auto deprecated(PA parser) {
  return DeprecatedParser<LF, PA>(parser);
}

// Parsing objects with "source" members.
template <typename PA> class SourcedParser {
public:
  using resultType = typename PA::resultType;
  constexpr SourcedParser(const SourcedParser &) = default;
  constexpr SourcedParser(PA parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    const char *start{state.GetLocation()};
    auto result{parser_.Parse(state)};
    if (result) {
      const char *end{state.GetLocation()};
      for (; start < end && start[0] == ' '; ++start) {
      }
      for (; start < end && end[-1] == ' '; --end) {
      }
      result->source = CharBlock{start, end};
    }
    return result;
  }

private:
  const PA parser_;
};

template <typename PA> inline constexpr auto sourced(PA parser) {
  return SourcedParser<PA>{parser};
}
} // namespace Fortran::parser
#endif // FORTRAN_PARSER_BASIC_PARSERS_H_
