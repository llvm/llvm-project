//===-- lib/Evaluate/character-value-impl.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "character-value-impl.h"
#include "flang/Common/idioms.h"
#include "flang/Evaluate/common.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cstring>

namespace Fortran::evaluate::value {

CharacterValueImpl::CharacterValueImpl(
    KindsEnum kind, std::size_t n, char32_t c) {
  withCharProto(kind, [this, n, c](auto ct) {
    using CharT = std::decay_t<decltype(ct)>;
    storage_ = std::basic_string<CharT>(n, static_cast<CharT>(c));
  });
}

CharacterValueImpl CharacterValueImpl::Zero(KindsEnum kind) {
  return withCharProto(kind, [kind](auto c) {
    using Char = std::decay_t<decltype(c)>;
    return CharacterValueImpl{kind, std::basic_string<Char>{}};
  });
}

CharacterValueImpl CharacterValueImpl::FromRawBytes(
    KindsEnum kind, const void *raw, size_t size) {
  return withCharProto(kind, [kind, raw, size](auto charProto) {
    using CharT = decltype(charProto);
    CHECK(size % sizeof(CharT) == 0);
    std::basic_string<CharT> s;
    if (size > 0) {
      s.assign(static_cast<const CharT *>(raw), size / sizeof(CharT));
    }
    return CharacterValueImpl{kind, std::move(s)};
  });
}

void CharacterValueImpl::print(llvm::raw_ostream &os) const {
  os << static_cast<int>(kind()) << '_';
  withStdString(
      [&](const auto &s) { os << parser::QuoteCharacterLiteral(s, true); });
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void CharacterValueImpl::dump() const {
  print(llvm::errs());
  llvm::errs() << '\n';
}
#endif

std::size_t CharacterValueImpl::charSize() const {
  return common::visit(
      [](const auto &s) -> std::size_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          llvm_unreachable("operation not supported on uninitialized value");
        } else {
          return sizeof(typename std::decay_t<decltype(s)>::value_type);
        }
      },
      storage_);
}

std::size_t CharacterValueImpl::size() const {
  return common::visit(
      [](const auto &s) -> std::size_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          return 0;
        } else {
          return s.size();
        }
      },
      storage_);
}

void *CharacterValueImpl::charData() {
  return common::visit(
      [](auto &s) -> void * {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          // No data available in monostate
          return nullptr;
        } else {
          return static_cast<void *>(s.data());
        }
      },
      storage_);
}

const void *CharacterValueImpl::charData() const {
  return common::visit(
      [](const auto &s) -> const void * {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          // No data available in monostate
          return nullptr;
        } else {
          return static_cast<const void *>(s.data());
        }
      },
      storage_);
}

Ordering CharacterValueImpl::Compare(const CharacterValueImpl &y) const {
  return common::visit(
      [](const auto &xs, const auto &ys) -> Ordering {
        using XS = std::decay_t<decltype(xs)>;
        using YS = std::decay_t<decltype(ys)>;

        // monostate represents an empty string of any type; here it is
        // polymorhpic to what it is compared to
        if constexpr (std::is_same_v<XS, YS>) {
          return Fortran::evaluate::Compare(xs, ys);
        } else if constexpr (std::is_same_v<XS, std::monostate> &&
            !std::is_same_v<YS, std::monostate>) {
          return Fortran::evaluate::Compare(YS{}, ys);
        } else if constexpr (!std::is_same_v<XS, std::monostate> &&
            std::is_same_v<YS, std::monostate>) {
          return Fortran::evaluate::Compare(xs, XS{});
        } else {
          llvm_unreachable("character comparison across differing kinds");
        }
      },
      this->storage_, y.storage_);
}

bool CharacterValueImpl::operator<(const CharacterValueImpl &y) const {
  return common::visit(
      [](const auto &xs, const auto &ys) -> bool {
        using XS = std::decay_t<decltype(xs)>;
        using YS = std::decay_t<decltype(ys)>;

        // monostate represents an empty string of any type; here it is
        // polymorphic to what it is compared to
        if constexpr (std::is_same_v<XS, YS>) {
          return xs < ys;
        } else if constexpr (std::is_same_v<XS, std::monostate> &&
            !std::is_same_v<YS, std::monostate>) {
          return YS{} < ys;
        } else if constexpr (!std::is_same_v<XS, std::monostate> &&
            std::is_same_v<YS, std::monostate>) {
          return xs < XS{};
        } else {
          llvm_unreachable("character comparison across differing kinds");
        }
      },
      this->storage_, y.storage_);
}

bool CharacterValueImpl::operator==(const CharacterValueImpl &y) const {
  return common::visit(
      [](const auto &xs, const auto &ys) -> bool {
        using XS = std::decay_t<decltype(xs)>;
        using YS = std::decay_t<decltype(ys)>;

        // monostate represents an empty string of any type; here it is
        // polymorhpic to what it is compared to
        if constexpr (std::is_same_v<XS, YS>) {
          return xs == ys;
        } else if constexpr (std::is_same_v<XS, std::monostate> &&
            !std::is_same_v<YS, std::monostate>) {
          return YS{} == ys;
        } else if constexpr (!std::is_same_v<XS, std::monostate> &&
            std::is_same_v<YS, std::monostate>) {
          return xs == XS{};
        } else {
          llvm_unreachable("character comparison across differing kinds");
        }
      },
      this->storage_, y.storage_);
}

void CharacterValueImpl::assign(KindsEnum kind, std::size_t n, char32_t c) {
  return withCharProto(kind, [this, n, c](auto ct) {
    using CharT = decltype(ct);
    storage_ = std::basic_string<CharT>(n, static_cast<CharT>(c));
  });
}

void CharacterValueImpl::erase(std::size_t pos) {
  common::visit(
      [pos](auto &s) {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          llvm_unreachable("operation not supported on uninitialized value");
        } else {
          s.erase(pos);
        }
      },
      storage_);
}

void CharacterValueImpl::append(std::size_t n, char32_t c) {
  common::visit(
      [n, c](auto &s) {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          llvm_unreachable("operation not supported on uninitialized value");
        } else {
          using CharT = typename std::decay_t<decltype(s)>::value_type;
          s.append(n, static_cast<CharT>(c));
        }
      },
      storage_);
}

CharacterValueImpl &CharacterValueImpl::replace(
    std::size_t pos, std::size_t len, const CharacterValueImpl &other) {
  common::visit(
      [pos, len](auto &s, const auto &o) {
        if constexpr (!std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate> &&
            !std::is_same_v<std::decay_t<decltype(o)>, std::monostate> &&
            std::is_same_v<std::decay_t<decltype(s)>,
                std::decay_t<decltype(o)>>) {
          s.replace(pos, len, o);
        } else {
          llvm_unreachable("operation not supported on uninitialized value or "
                           "values of different kinds");
        }
      },
      storage_, other.storage_);
  return *this;
}

CharacterValueImpl CharacterValueImpl::substr(std::size_t pos) const {
  return common::visit(
      [pos](const auto &s) -> CharacterValueImpl {
        using StringT = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<StringT, std::monostate>) {
          llvm_unreachable("operation not supported on uninitialized value");
        } else {
          return CharacterValueImpl{
              static_cast<KindsEnum>(sizeof(typename StringT::value_type)),
              s.substr(pos)};
        }
      },
      storage_);
}

CharacterValueImpl CharacterValueImpl::substr(
    std::size_t pos, std::size_t len) const {
  return common::visit(
      [pos, len](const auto &s) -> CharacterValueImpl {
        using StringT = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          llvm_unreachable("operation not supported on uninitialized value");
        } else {
          return CharacterValueImpl{
              static_cast<KindsEnum>(sizeof(typename StringT::value_type)),
              s.substr(pos, len)};
        }
      },
      storage_);
}

std::optional<llvm::StringRef> CharacterValueImpl::AsStringRef() const {
  if (IsMonostate()) {
    return llvm::StringRef{};
  }
  if (const auto *s{std::get_if<std::string>(&storage_)}) {
    return *s;
  }
  return std::nullopt;
}

/// Return the string as std::string if kind==1, or nullopt otherwise.
std::optional<std::string> CharacterValueImpl::AsStdString() const {
  if (IsMonostate()) {
    return std::string{};
  }

  if (const auto *s{std::get_if<std::string>(&storage_)}) {
    return *s;
  } else {
    return std::nullopt;
  }
}

std::optional<std::u16string> CharacterValueImpl::AsU16String() const {
  if (IsMonostate()) {
    return std::u16string{};
  }

  if (const auto *s{std::get_if<std::u16string>(&storage_)}) {
    return *s;
  } else {
    return std::nullopt;
  }
}

std::optional<std::u32string> CharacterValueImpl::AsU32String() const {
  if (IsMonostate()) {
    return std::u32string{};
  }

  if (const auto *s{std::get_if<std::u32string>(&storage_)}) {
    return *s;
  } else {
    return std::nullopt;
  }
}

std::string CharacterValueImpl::ToStdString() const {
  return common::visit(
      [](const auto &s) {
        using StringT = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<StringT, std::monostate>) {
          return std::string{};
        } else if constexpr (std::is_same_v<StringT, std::string>) {
          return s;
        } else {
          std::string result(s.size(), '\0');
          for (auto [i, c] : llvm::enumerate(s)) {
            result[i] = c;
          }
          return result;
        }
      },
      storage_);
}

CharacterValueImpl CharacterValueImpl::ToAscii(KindsEnum kind) const {
  if (IsMonostate()) {
    return Zero(kind);
  }

  return withStdString([kind](const auto &s) -> CharacterValueImpl {
    return withCharProto(kind, [&s](auto ct) -> CharacterValueImpl {
      using CharT = std::decay_t<decltype(ct)>;
      using StringT = std::basic_string<CharT>;
      // Fortran character conversion is well defined between distinct kinds
      // only when the actual characters are valid 7-bit ASCII.
      StringT str;
      for (auto iter{s.cbegin()}; iter != s.cend(); ++iter) {
        if (static_cast<std::uint64_t>(*iter) > 127) {
          return Zero(static_cast<KindsEnum>(sizeof(ct)));
        }
        str.push_back(static_cast<CharT>(*iter));
      }
      return CharacterValueImpl{static_cast<KindsEnum>(sizeof(CharT)), str};
    });
  });
}

void CharacterValueImpl::reserve(std::size_t n) {
  common::visit(
      [n](auto &s) {
        if constexpr (!std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          s.reserve(n);
        }
      },
      storage_);
}

char32_t CharacterValueImpl::operator[](std::size_t i) const {
  return common::visit(
      [i](const auto &s) -> char32_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          llvm_unreachable("operation not supported on uninitialized value");
        } else {
          return static_cast<char32_t>(s[i]);
        }
        return 0;
      },
      storage_);
}

CharacterValueImpl CharacterValueImpl::operator+(
    const CharacterValueImpl &y) const {
  return common::visit(
      [](const auto &a, const auto &b) -> CharacterValueImpl {
        if constexpr (std::is_same_v<std::decay_t<decltype(a)>,
                          std::decay_t<decltype(b)>> &&
            !std::is_same_v<std::decay_t<decltype(a)>, std::monostate>) {
          using StringT = std::decay_t<decltype(a)>;
          return CharacterValueImpl{
              static_cast<KindsEnum>(sizeof(typename StringT::value_type)),
              a + b};
        } else {
          llvm_unreachable("operation not supported on uninitialized value or "
                           "values of different kinds");
        }
        return CharacterValueImpl{};
      },
      storage_, y.storage_);
}

CharacterValueImpl &CharacterValueImpl::operator+=(
    const CharacterValueImpl &y) {
  common::visit(
      [](auto &a, const auto &b) {
        if constexpr (std::is_same_v<std::decay_t<decltype(a)>,
                          std::decay_t<decltype(b)>> &&
            !std::is_same_v<std::decay_t<decltype(a)>, std::monostate>) {
          a += b;
        } else {
          llvm_unreachable("operation not supported on uninitialized value or "
                           "values of different kinds");
        }
      },
      storage_, y.storage_);
  return *this;
}

CharacterValueImpl &CharacterValueImpl::operator+=(char c) {
  common::visit(
      [c](auto &s) {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          llvm_unreachable("operation not supported on uninitialized value");
        } else {
          using CharT = typename std::decay_t<decltype(s)>::value_type;
          s.push_back(static_cast<CharT>(c));
        }
      },
      storage_);
  return *this;
}

std::size_t CharacterValueImpl::find_first_not_of(char32_t c) const {
  return common::visit(
      [c](const auto &s) -> std::size_t {
        if constexpr (!std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          using CharT = typename std::decay_t<decltype(s)>::value_type;
          return s.find_first_not_of(static_cast<CharT>(c));
        } else {
          llvm_unreachable("Unsupported combination of character kinds");
          return std::string::npos;
        }
      },
      storage_);
}

std::size_t CharacterValueImpl::find_last_not_of(char32_t c) const {
  return common::visit(
      [c](const auto &s) -> std::size_t {
        if constexpr (!std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          using CharT = typename std::decay_t<decltype(s)>::value_type;
          return s.find_last_not_of(static_cast<CharT>(c));
        } else {
          llvm_unreachable("Unsupported combination of character kinds");
          return std::string::npos;
        }
      },
      storage_);
}

std::size_t CharacterValueImpl::find_first_not_of(
    const CharacterValueImpl &set) const {
  return common::visit(
      [](const auto &s, const auto &p) -> std::size_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          // Nothing to find in an empty string
          return std::string::npos;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                                 std::decay_t<decltype(p)>> &&
            !std::is_same_v<std::decay_t<decltype(s)>, std::monostate>) {
          return s.find_first_not_of(p);
        } else {
          llvm_unreachable("Unsupported combination of character kinds");
          return std::string::npos;
        }
      },
      storage_, set.storage_);
}

std::size_t CharacterValueImpl::find_last_not_of(
    const CharacterValueImpl &set) const {
  return common::visit(
      [](const auto &s, const auto &p) -> std::size_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          // Nothing to find in an empty string
          return std::string::npos;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                                 std::decay_t<decltype(p)>> &&
            !std::is_same_v<std::decay_t<decltype(s)>, std::monostate>) {
          return s.find_last_not_of(p);
        } else {
          llvm_unreachable("Unsupported combination of character kinds");
          return std::string::npos;
        }
      },
      storage_, set.storage_);
}

std::size_t CharacterValueImpl::find(const CharacterValueImpl &pattern) const {
  return common::visit(
      [](const auto &s, const auto &p) -> std::size_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(p)>,
                          std::monostate>) {
          // Empty string always matches beginning
          return 0;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                                 std::monostate>) {
          // Nothing to find in an empty string, unless the pattern is itself an
          // empty string
          return p.empty() ? 0 : std::string::npos;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                                 std::decay_t<decltype(p)>> &&
            !std::is_same_v<std::decay_t<decltype(s)>, std::monostate>) {
          return s.find(p);
        } else {
          llvm_unreachable("Unsupported combination of character kinds");
          return std::string::npos;
        }
      },
      storage_, pattern.storage_);
}

std::size_t CharacterValueImpl::rfind(const CharacterValueImpl &pattern) const {
  return common::visit(
      [](const auto &s, const auto &p) -> std::size_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          // Nothing to find in an empty string
          return std::string::npos;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                                 std::decay_t<decltype(p)>> &&
            !std::is_same_v<std::decay_t<decltype(s)>, std::monostate>) {
          return s.rfind(p);
        }
        llvm_unreachable("Unsupported combination of character kinds");
        return std::string::npos;
      },
      storage_, pattern.storage_);
}

std::size_t CharacterValueImpl::find_first_of(
    const CharacterValueImpl &set) const {
  return common::visit(
      [](const auto &s, const auto &p) -> std::size_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          // Nothing to find in an empty string
          return std::string::npos;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                                 std::decay_t<decltype(p)>> &&
            !std::is_same_v<std::decay_t<decltype(s)>, std::monostate>) {
          return s.find_first_of(p);
        } else {
          llvm_unreachable("Unsupported combination of character kinds");
          return std::string::npos;
        }
      },
      storage_, set.storage_);
}

std::size_t CharacterValueImpl::find_last_of(
    const CharacterValueImpl &set) const {
  return common::visit(
      [](const auto &s, const auto &p) -> std::size_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          // Nothing to find in an empty string
          return std::string::npos;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                                 std::decay_t<decltype(p)>> &&
            !std::is_same_v<std::decay_t<decltype(s)>, std::monostate>) {
          return s.find_last_of(p);
        } else {
          llvm_unreachable("Unsupported combination of character kinds");
          return std::string::npos;
        }
      },
      storage_, set.storage_);
}

void CharacterValueImpl::StoreRawBytes(
    void *dst, std::size_t size, bool *changed) const {
  common::visit(
      [&](const auto &word) {
        if constexpr (std::is_same_v<std::decay_t<decltype(word)>,
                          std::monostate>) {
          CHECK(size == 0);
          // Nothing to store
        } else {
          using Character = std::decay_t<decltype(word)>;
          using CharT = typename Character::value_type;
          CHECK(size % sizeof(CharT) == 0);
          if (size > 0) {
            std::size_t payloadSize{
                std::min(size, sizeof(CharT) * word.size())};
            std::size_t padSize{size - payloadSize};

            Character strWithPadding{word};
            strWithPadding.append(
                padSize / sizeof(CharT), static_cast<CharT>(' '));

            if (changed) {
              if (std::memcmp(dst, strWithPadding.data(), size) == 0) {
                return;
              }
              *changed = true;
            }
            std::memcpy(dst, strWithPadding.data(), size);
          }
        }
      },
      storage_);
}

} // namespace Fortran::evaluate::value
