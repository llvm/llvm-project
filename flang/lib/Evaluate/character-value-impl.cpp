//===-- lib/Evaluate/character-value-impl.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/character-value-impl.h"
#include "flang/Common/idioms.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cstring>

namespace Fortran::evaluate::value {

int CharacterValueImpl::kind() const {
  switch (storage_.index()) {
  case 1:
    return 1;
  case 2:
    return 2;
  case 3:
    return 4;
  default:
    llvm_unreachable("unspecified char kind");
  }
}

int CharacterValueImpl::bits() const { return 8 * kind(); }

CharacterValueImpl CharacterValueImpl::Zero(int kind) {
  switch (kind) {
  case 2:
    return CharacterValueImpl{std::u16string{}};
  case 4:
    return CharacterValueImpl{std::u32string{}};
  case 1:
    return CharacterValueImpl{std::string{}};
  default:
    llvm_unreachable("unsupported char kind");
  }
}

bool CharacterValueImpl::IsZero() const {
  if (IsMonostate()) {
    return true;
  }
  return std::visit(
      [](const auto &s) -> bool {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          return true;
        } else if (s.empty()) {
          return true;
        } else {
          using CharT = typename std::decay_t<decltype(s)>::value_type;
          return std::all_of(
              s.begin(), s.end(), [](CharT c) { return c == CharT{}; });
        }
      },
      storage_);
}

bool CharacterValueImpl::StoreRawBytes(void *to, std::size_t bytes) const {
  return std::visit(
      [&](const auto &s) -> bool {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          if (bytes == 0) {
            return false;
          }
          if (std::all_of(static_cast<const char *>(to),
                  static_cast<const char *>(to) + bytes,
                  [](char x) { return x == 0; })) {
            return false;
          }
          std::memset(to, 0, bytes);
          return true;
        } else {
          std::size_t payloadBytes{std::min(bytes,
              s.size() *
                  sizeof(typename std::decay_t<decltype(s)>::value_type))};
          if (std::memcmp(to, s.data(), payloadBytes) != 0 ||
              (payloadBytes < bytes &&
                  !std::all_of(
                      static_cast<const char *>(to) + payloadBytes,
                      static_cast<const char *>(to) + bytes,
                      [](char x) { return x == 0; }))) {
            std::memcpy(to, s.data(), payloadBytes);
            if (payloadBytes < bytes) {
              std::memset(static_cast<char *>(to) + payloadBytes, 0,
                  bytes - payloadBytes);
            }
            return true;
          }
          return false;
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

void CharacterValueImpl::erase(std::size_t pos) {
  std::visit(
      [pos](auto &s) {
        if constexpr (!std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          s.erase(pos);
        } else {
          llvm_unreachable("operation not supported on uninitialized value");
        }
      },
      storage_);
}

void CharacterValueImpl::append(std::size_t n, char c) {
  std::visit(
      [n, c](auto &s) {
        if constexpr (!std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          using CharT = typename std::decay_t<decltype(s)>::value_type;
          s.append(n, static_cast<CharT>(c));
        } else {
          llvm_unreachable("operation not supported on uninitialized value");
        }
      },
      storage_);
}

void CharacterValueImpl::append(std::size_t n, char16_t c) {
  std::visit(
      [n, c](auto &s) {
        if constexpr (!std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          using CharT = typename std::decay_t<decltype(s)>::value_type;
          s.append(n, static_cast<CharT>(c));
        } else {
          llvm_unreachable("operation not supported on uninitialized value");
        }
      },
      storage_);
}

void CharacterValueImpl::append(std::size_t n, char32_t c) {
  std::visit(
      [n, c](auto &s) {
        if constexpr (!std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          using CharT = typename std::decay_t<decltype(s)>::value_type;
          s.append(n, static_cast<CharT>(c));
        } else {
          llvm_unreachable("operation not supported on uninitialized value");
        }
      },
      storage_);
}

CharacterValueImpl &CharacterValueImpl::replace(
    std::size_t pos, std::size_t len, const CharacterValueImpl &other) {
  std::visit(
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
  return std::visit(
      [pos](const auto &s) -> CharacterValueImpl {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          llvm_unreachable("operation not supported on uninitialized value");
        } else {
          return CharacterValueImpl{s.substr(pos)};
        }
      },
      storage_);
}

CharacterValueImpl CharacterValueImpl::substr(
    std::size_t pos, std::size_t len) const {
  return std::visit(
      [pos, len](const auto &s) -> CharacterValueImpl {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          llvm_unreachable("operation not supported on uninitialized value");
        } else {
          return CharacterValueImpl{s.substr(pos, len)};
        }
      },
      storage_);
}

std::optional<std::string> CharacterValueImpl::ToStdString() const {
  if (const auto *s{std::get_if<std::string>(&storage_)}) {
    return *s;
  }
  return std::nullopt;
}

void CharacterValueImpl::reserve(std::size_t n) {
  std::visit(
      [n](auto &s) {
        if constexpr (!std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          s.reserve(n);
        }
      },
      storage_);
}

char32_t CharacterValueImpl::operator[](std::size_t i) const {
  return std::visit(
      [i](const auto &s) -> char32_t {
        if constexpr (!std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          return static_cast<char32_t>(s[i]);
        } else {
          llvm_unreachable("operation not supported on uninitialized value");
        }
        return 0;
      },
      storage_);
}

bool CharacterValueImpl::operator<(const CharacterValueImpl &y) const {
  return std::visit(
      [](const auto &a, const auto &b) -> bool {
        if constexpr (std::is_same_v<std::decay_t<decltype(a)>,
                          std::decay_t<decltype(b)>> &&
            !std::is_same_v<std::decay_t<decltype(a)>, std::monostate>) {
          return a < b;
        } else {
          llvm_unreachable("operation not supported on uninitialized value or "
                           "values of different kinds");
        }
        return false;
      },
      storage_, y.storage_);
}

CharacterValueImpl CharacterValueImpl::operator+(
    const CharacterValueImpl &y) const {
  return std::visit(
      [](const auto &a, const auto &b) -> CharacterValueImpl {
        if constexpr (std::is_same_v<std::decay_t<decltype(a)>,
                          std::decay_t<decltype(b)>> &&
            !std::is_same_v<std::decay_t<decltype(a)>, std::monostate>) {
          return CharacterValueImpl{a + b};
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
  std::visit(
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
  std::visit(
      [c](auto &s) {
        if constexpr (!std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          using CharT = typename std::decay_t<decltype(s)>::value_type;
          s.push_back(static_cast<CharT>(c));
        } else {
          llvm_unreachable("operation not supported on uninitialized value");
        }
      },
      storage_);
  return *this;
}

std::size_t CharacterValueImpl::find_first_not_of(
    const CharacterValueImpl &set) const {
  return std::visit(
      [](const auto &s, const auto &p) -> std::size_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::decay_t<decltype(p)>> &&
            !std::is_same_v<std::decay_t<decltype(s)>, std::monostate>) {
          return s.find_first_not_of(p);
        }
        return std::string::npos;
      },
      storage_, set.storage_);
}

std::size_t CharacterValueImpl::find_last_not_of(
    const CharacterValueImpl &set) const {
  return std::visit(
      [](const auto &s, const auto &p) -> std::size_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::decay_t<decltype(p)>> &&
            !std::is_same_v<std::decay_t<decltype(s)>, std::monostate>) {
          return s.find_last_not_of(p);
        }
        return std::string::npos;
      },
      storage_, set.storage_);
}

std::size_t CharacterValueImpl::find(const CharacterValueImpl &pattern) const {
  return std::visit(
      [](const auto &s, const auto &p) -> std::size_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::decay_t<decltype(p)>> &&
            !std::is_same_v<std::decay_t<decltype(s)>, std::monostate>) {
          return s.find(p);
        }
        return std::string::npos;
      },
      storage_, pattern.storage_);
}

std::size_t CharacterValueImpl::rfind(const CharacterValueImpl &pattern) const {
  return std::visit(
      [](const auto &s, const auto &p) -> std::size_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::decay_t<decltype(p)>> &&
            !std::is_same_v<std::decay_t<decltype(s)>, std::monostate>) {
          return s.rfind(p);
        }
        return std::string::npos;
      },
      storage_, pattern.storage_);
}

std::size_t CharacterValueImpl::find_first_of(
    const CharacterValueImpl &set) const {
  return std::visit(
      [](const auto &s, const auto &p) -> std::size_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::decay_t<decltype(p)>> &&
            !std::is_same_v<std::decay_t<decltype(s)>, std::monostate>) {
          return s.find_first_of(p);
        }
        return std::string::npos;
      },
      storage_, set.storage_);
}

std::size_t CharacterValueImpl::find_last_of(
    const CharacterValueImpl &set) const {
  return std::visit(
      [](const auto &s, const auto &p) -> std::size_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::decay_t<decltype(p)>> &&
            !std::is_same_v<std::decay_t<decltype(s)>, std::monostate>) {
          return s.find_last_of(p);
        }
        return std::string::npos;
      },
      storage_, set.storage_);
}

std::size_t CharacterValueImpl::charSize() const {
  return std::visit(
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

void *CharacterValueImpl::charData() {
  return std::visit(
      [](auto &s) -> void * {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          return nullptr;
        } else {
          return static_cast<void *>(s.data());
        }
      },
      storage_);
}

const void *CharacterValueImpl::charData() const {
  return std::visit(
      [](const auto &s) -> const void * {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          return nullptr;
        } else {
          return static_cast<const void *>(s.data());
        }
      },
      storage_);
}

std::size_t CharacterValueImpl::find_first_not_of_char(char32_t c) const {
  return std::visit(
      [c](const auto &s) -> std::size_t {
        if constexpr (!std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          using CharT = typename std::decay_t<decltype(s)>::value_type;
          //  assert(static_cast<CharT>(c) == c);
          return s.find_first_not_of(static_cast<CharT>(c));
        }
        return std::string::npos;
      },
      storage_);
}

std::size_t CharacterValueImpl::find_last_not_of_char(char32_t c) const {
  return std::visit(
      [c](const auto &s) -> std::size_t {
        if constexpr (!std::is_same_v<std::decay_t<decltype(s)>,
                          std::monostate>) {
          using CharT = typename std::decay_t<decltype(s)>::value_type;
          //  assert(static_cast<CharT>(c) == c);
          return s.find_last_not_of(static_cast<CharT>(c));
        }
        return std::string::npos;
      },
      storage_);
}

} // namespace Fortran::evaluate::value
