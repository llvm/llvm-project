//===-- lib/Evaluate/character-value.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/character-value.h"
#include "flang/Evaluate/character-value-impl.h"
#include "flang/Evaluate/common.h"
#include "llvm/Support/ErrorHandling.h"
#include <new>
#include <string>

namespace Fortran::evaluate::value {

static_assert(sizeof(CharacterValueImpl) == detail::kCharacterObjectSize);
static_assert(alignof(CharacterValueImpl) <= detail::kCharacterObjectAlign);
static_assert(sizeof(CharacterValue) == sizeof(CharacterValueImpl));

CharacterValue::CharacterValue() { new (this) CharacterValueImpl(); }
CharacterValue::~CharacterValue() { value().~CharacterValueImpl(); }
CharacterValue::CharacterValue(const CharacterValue &x) {
  new (this) CharacterValueImpl(x.value());
}
CharacterValue::CharacterValue(CharacterValue &&x) {
  new (this) CharacterValueImpl(std::move(x.value()));
}
CharacterValue &CharacterValue::operator=(const CharacterValue &x) {
  value() = x.value();
  return *this;
}
CharacterValue &CharacterValue::operator=(CharacterValue &&x) {
  value() = std::move(x.value());
  return *this;
}

CharacterValue::CharacterValue(std::string s) {
  new (this) CharacterValueImpl(std::move(s));
}
CharacterValue::CharacterValue(std::u16string s) {
  new (this) CharacterValueImpl(std::move(s));
}
CharacterValue::CharacterValue(std::u32string s) {
  new (this) CharacterValueImpl(std::move(s));
}
CharacterValue::CharacterValue(std::size_t n, char c) {
  new (this) CharacterValueImpl(n, c);
}
CharacterValue::CharacterValue(std::size_t n, char16_t c) {
  new (this) CharacterValueImpl(n, c);
}
CharacterValue::CharacterValue(std::size_t n, char32_t c) {
  new (this) CharacterValueImpl(n, c);
}

bool CharacterValue::operator==(const CharacterValue &y) const {
  return value() == y.value();
}
bool CharacterValue::operator<(const CharacterValue &y) const {
  return value() < y.value();
}
int CharacterValue::kind() const { return value().kind(); }
int CharacterValue::bits() const { return value().bits(); }
bool CharacterValue::IsMonostate() const { return value().IsMonostate(); }
bool CharacterValue::IsZero() const { return value().IsZero(); }
CharacterValue CharacterValue::Zero(int kind) {
  CharacterValue result;
  result.value() = CharacterValueImpl::Zero(kind);
  return result;
}

std::size_t CharacterValue::size() const { return value().size(); }
void CharacterValue::assign(std::size_t n, char c) { value().assign(n, c); }
void CharacterValue::assign(std::size_t n, char16_t c) { value().assign(n, c); }
void CharacterValue::assign(std::size_t n, char32_t c) { value().assign(n, c); }
void CharacterValue::assign(const char *p, std::size_t n) {
  value().assign(p, n);
}
void CharacterValue::assign(const char16_t *p, std::size_t n) {
  value().assign(p, n);
}
void CharacterValue::assign(const char32_t *p, std::size_t n) {
  value().assign(p, n);
}
void CharacterValue::erase(std::size_t pos) { value().erase(pos); }
void CharacterValue::append(std::size_t n, char c) { value().append(n, c); }
void CharacterValue::append(std::size_t n, char16_t c) { value().append(n, c); }
void CharacterValue::append(std::size_t n, char32_t c) { value().append(n, c); }
CharacterValue &CharacterValue::replace(
    std::size_t pos, std::size_t len, const CharacterValue &other) {
  value().replace(pos, len, other.value());
  return *this;
}
CharacterValue CharacterValue::substr(std::size_t pos) const {
  CharacterValue result;
  result.value() = value().substr(pos);
  return result;
}
CharacterValue CharacterValue::substr(std::size_t pos, std::size_t len) const {
  CharacterValue result;
  result.value() = value().substr(pos, len);
  return result;
}
std::optional<std::string> CharacterValue::ToStdString() const {
  return value().ToStdString();
}
void CharacterValue::reserve(std::size_t n) { value().reserve(n); }
char32_t CharacterValue::operator[](std::size_t i) const { return value()[i]; }
CharacterValue CharacterValue::operator+(const CharacterValue &y) const {
  CharacterValue result;
  result.value() = value() + y.value();
  return result;
}
CharacterValue &CharacterValue::operator+=(const CharacterValue &y) {
  value() += y.value();
  return *this;
}
CharacterValue &CharacterValue::operator+=(char c) {
  value() += c;
  return *this;
}
std::size_t CharacterValue::find_first_not_of(const CharacterValue &set) const {
  return value().find_first_not_of(set.value());
}
std::size_t CharacterValue::find_last_not_of(const CharacterValue &set) const {
  return value().find_last_not_of(set.value());
}
std::size_t CharacterValue::find(const CharacterValue &pattern) const {
  return value().find(pattern.value());
}
std::size_t CharacterValue::rfind(const CharacterValue &pattern) const {
  return value().rfind(pattern.value());
}
std::size_t CharacterValue::find_first_of(const CharacterValue &set) const {
  return value().find_first_of(set.value());
}
std::size_t CharacterValue::find_last_of(const CharacterValue &set) const {
  return value().find_last_of(set.value());
}
std::size_t CharacterValue::charSize() const { return value().charSize(); }
const void *CharacterValue::data() const { return value().data(); }
void *CharacterValue::charData() { return value().charData(); }
const void *CharacterValue::charData() const { return value().charData(); }
std::size_t CharacterValue::find_first_not_of_char(char32_t c) const {
  return value().find_first_not_of(c);
}
std::size_t CharacterValue::find_last_not_of_char(char32_t c) const {
  return value().find_last_not_of(c);
}

} // namespace Fortran::evaluate::value

namespace Fortran::evaluate {

// Fortran CHARACTER comparison: the shorter operand is blank-padded to the
// length of the longer before comparing.  We recover the active string
// alternative via WithChar and defer to the std::basic_string overload of
// Compare in common.h, which performs the blank padding.
Ordering Compare(
    const value::CharacterValue &x, const value::CharacterValue &y) {
  if (x.IsMonostate() || y.IsMonostate()) {
    return Ordering::Equal;
  }
  return x.value().WithChar([&y](const auto &xs) -> Ordering {
    using XS = std::decay_t<decltype(xs)>;
    return y.value().WithChar([&xs](const auto &ys) -> Ordering {
      if constexpr (std::is_same_v<XS, std::decay_t<decltype(ys)>>) {
        return Compare(xs, ys);
      } else {
        // Same-KIND operands always share a storage type; differing types
        // cannot arise from a well-typed relational expression.
        llvm_unreachable("character comparison across differing kinds");
      }
    });
  });
}

} // namespace Fortran::evaluate
