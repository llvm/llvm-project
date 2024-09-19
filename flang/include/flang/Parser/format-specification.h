//===-- include/flang/Parser/format-specification.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_FORMAT_SPECIFICATION_H_
#define FORTRAN_PARSER_FORMAT_SPECIFICATION_H_

// Represent parses of Fortran format specifications from FORMAT statements
// and character literals in formatted I/O statements at compilation time
// as well as from character variables and expressions at run time.
// From requirement productions R1302-R1321 of the Fortran 2018 draft
// standard (q.v.), extended with Hollerith.  These structures have been
// isolated so that they may be used in the run-time without introducing
// dependences on other parts of the compiler's source code.
// TODO: support Q formatting extension?

#include "flang/Common/idioms.h"
#include <cinttypes>
#include <list>
#include <optional>
#include <string>
#include <variant>

namespace Fortran::format {

// R1307 data-edit-desc (part 1 of 2) ->
//         I w [. m] | B w [. m] | O w [. m] | Z w [. m] | F w . d |
//         E w . d [E e] | EN w . d [E e] | ES w . d [E e] | EX w . d [E e] |
//         G w [. d [E e]] | L w | A [w] | D w . d
// R1308 w -> digit-string
// R1309 m -> digit-string
// R1310 d -> digit-string
// R1311 e -> digit-string
struct IntrinsicTypeDataEditDesc {
  enum class Kind { I, B, O, Z, F, E, EN, ES, EX, G, L, A, D };
  IntrinsicTypeDataEditDesc() = delete;
  IntrinsicTypeDataEditDesc(IntrinsicTypeDataEditDesc &&) = default;
  IntrinsicTypeDataEditDesc &operator=(IntrinsicTypeDataEditDesc &&) = default;
  IntrinsicTypeDataEditDesc(Kind &&k, std::optional<int> &&w,
      std::optional<int> &&d, std::optional<int> &&e)
      : kind{k}, width{std::move(w)}, digits{std::move(d)}, exponentWidth{
                                                                std::move(e)} {}
  Kind kind;
  std::optional<int> width; // w
  std::optional<int> digits; // m or d
  std::optional<int> exponentWidth; // e
};

// R1307 data-edit-desc (part 2 of 2) ->
//         DT [char-literal-constant] [( v-list )]
// R1312 v -> [sign] digit-string
struct DerivedTypeDataEditDesc {
  DerivedTypeDataEditDesc() = delete;
  DerivedTypeDataEditDesc(DerivedTypeDataEditDesc &&) = default;
  DerivedTypeDataEditDesc &operator=(DerivedTypeDataEditDesc &&) = default;
  DerivedTypeDataEditDesc(std::string &&t, std::list<std::int64_t> &&p)
      : type{std::move(t)}, parameters{std::move(p)} {}
  std::string type;
  std::list<std::int64_t> parameters;
};

// R1313 control-edit-desc ->
//         position-edit-desc | [r] / | : | sign-edit-desc | k P |
//         blank-interp-edit-desc | round-edit-desc | decimal-edit-desc |
//         @ \ | $
// R1315 position-edit-desc -> T n | TL n | TR n | n X
// R1316 n -> digit-string
// R1317 sign-edit-desc -> SS | SP | S
// R1318 blank-interp-edit-desc -> BN | BZ
// R1319 round-edit-desc -> RU | RD | RZ | RN | RC | RP
// R1320 decimal-edit-desc -> DC | DP
struct ControlEditDesc {
  enum class Kind {
    T,
    TL,
    TR,
    X,
    Slash,
    Colon,
    SS,
    SP,
    S,
    P,
    BN,
    BZ,
    RU,
    RD,
    RZ,
    RN,
    RC,
    RP,
    DC,
    DP,
    Dollar, // extension: inhibit newline on output
    Backslash, // ditto, but only on terminals
  };
  ControlEditDesc() = delete;
  ControlEditDesc(ControlEditDesc &&) = default;
  ControlEditDesc &operator=(ControlEditDesc &&) = default;
  explicit ControlEditDesc(Kind k) : kind{k} {}
  ControlEditDesc(Kind k, std::int64_t ct) : kind{k}, count{ct} {}
  ControlEditDesc(std::int64_t ct, Kind k) : kind{k}, count{ct} {}
  Kind kind;
  std::int64_t count{1}; // r, k, or n
};

// R1304 format-item ->
//         [r] data-edit-desc | control-edit-desc | char-string-edit-desc |
//         [r] ( format-items )
// R1306 r -> digit-string
// R1321 char-string-edit-desc
struct FormatItem {
  FormatItem() = delete;
  FormatItem(FormatItem &&) = default;
  FormatItem &operator=(FormatItem &&) = default;
  template <typename A, typename = common::NoLvalue<A>>
  FormatItem(std::optional<std::uint64_t> &&r, A &&x)
      : repeatCount{std::move(r)}, u{std::move(x)} {}
  template <typename A, typename = common::NoLvalue<A>>
  explicit FormatItem(A &&x) : u{std::move(x)} {}
  std::optional<std::uint64_t> repeatCount;
  std::variant<IntrinsicTypeDataEditDesc, DerivedTypeDataEditDesc,
      ControlEditDesc, std::string, std::list<FormatItem>>
      u;
};

// R1302 format-specification ->
//         ( [format-items] ) | ( [format-items ,] unlimited-format-item )
// R1303 format-items -> format-item [[,] format-item]...
// R1305 unlimited-format-item -> * ( format-items )
struct FormatSpecification {
  FormatSpecification() = delete;
  FormatSpecification(FormatSpecification &&) = default;
  FormatSpecification &operator=(FormatSpecification &&) = default;
  explicit FormatSpecification(std::list<FormatItem> &&is)
      : items(std::move(is)) {}
  FormatSpecification(std::list<FormatItem> &&is, std::list<FormatItem> &&us)
      : items(std::move(is)), unlimitedItems(std::move(us)) {}
  std::list<FormatItem> items, unlimitedItems;
};
} // namespace Fortran::format
#endif // FORTRAN_PARSER_FORMAT_SPECIFICATION_H_
