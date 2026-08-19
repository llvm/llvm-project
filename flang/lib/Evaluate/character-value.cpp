//===-- lib/Evaluate/character-value.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/character-value.h"
#include "character-value-impl.h"
#include "flang/Evaluate/common.h"
#include "llvm/Support/ErrorHandling.h"
#include <new>
#include <string>

namespace Fortran::evaluate::value {
static_assert(sizeof(CharacterValueImpl) == detail::kCharacterObjectSize);
static_assert(alignof(CharacterValueImpl) == detail::kCharacterObjectAlign);
static_assert(sizeof(CharacterValue) == sizeof(CharacterValueImpl));
static_assert(alignof(CharacterValue) == alignof(CharacterValueImpl));

CharacterValue::CharacterValue() { new (this) CharacterValueImpl(); }

CharacterValue::~CharacterValue() { impl().~CharacterValueImpl(); }

CharacterValue::CharacterValue(const CharacterValue &x) {
  new (this) CharacterValueImpl(x.impl());
}

CharacterValue::CharacterValue(CharacterValue &&x) {
  new (this) CharacterValueImpl(std::move(x.impl()));
}

CharacterValue &CharacterValue::operator=(const CharacterValue &x) {
  impl() = x.impl();
  return *this;
}

CharacterValue &CharacterValue::operator=(CharacterValue &&x) {
  impl() = std::move(x.impl());
  return *this;
}

CharacterValue::CharacterValue(KindsEnum kind, std::string s) {
  new (this) CharacterValueImpl(kind, std::move(s));
}

CharacterValue::CharacterValue(KindsEnum kind, std::u16string s) {
  CHECK(kind == KindsEnum::Kind2);
  new (this) CharacterValueImpl(kind, std::move(s));
}

CharacterValue::CharacterValue(KindsEnum kind, std::u32string s) {
  CHECK(kind == KindsEnum::Kind4);
  new (this) CharacterValueImpl(kind, std::move(s));
}

CharacterValue::CharacterValue(KindsEnum kind, std::size_t n, char32_t c) {
  new (this) CharacterValueImpl(kind, n, c);
}

CharacterValue CharacterValue::Zero(KindsEnum kind) {
  return FromImpl(CharacterValueImpl::Zero(kind));
}

CharacterValue CharacterValue::FromRawBytes(
    KindsEnum kind, const void *raw, size_t byteSize) {
  return FromImpl(CharacterValueImpl::FromRawBytes(kind, raw, byteSize));
}

void CharacterValue::print(llvm::raw_ostream &os) const { impl().print(os); }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void CharacterValue::dump() const { impl().dump(); }
#endif

bool CharacterValue::IsMonostate() const { return impl().IsMonostate(); }

bool CharacterValue::empty() const { return impl().empty(); }

std::size_t CharacterValue::size() const { return impl().size(); }

KindsEnum CharacterValue::kind() const { return impl().kind(); }

std::optional<llvm::StringRef> CharacterValue::AsStringRef() const {
  return impl().AsStringRef();
}

std::optional<std::u16string> CharacterValue::AsU16String() const {
  return impl().AsU16String();
}

std::optional<std::u32string> CharacterValue::AsU32String() const {
  return impl().AsU32String();
}

std::string CharacterValue::ToStdString() const { return impl().ToStdString(); }

Ordering CharacterValue::Compare(const CharacterValue &y) const {
  return impl().Compare(y.impl());
}

bool CharacterValue::operator<(const CharacterValue &y) const {
  return impl() < y.impl();
}

bool CharacterValue::operator==(const CharacterValue &y) const {
  return impl() == y.impl();
}

CharacterValue CharacterValue::ToAscii(KindsEnum kind) const {
  return FromImpl(impl().ToAscii(kind));
}

void CharacterValue::assign(KindsEnum kind, std::size_t n, char32_t c) {
  impl().assign(kind, n, c);
}

void CharacterValue::assign(const char *p, std::size_t n) {
  impl().assign(p, n);
}

void CharacterValue::assign(const char16_t *p, std::size_t n) {
  impl().assign(p, n);
}

void CharacterValue::assign(const char32_t *p, std::size_t n) {
  impl().assign(p, n);
}

void CharacterValue::erase(std::size_t pos) { impl().erase(pos); }

void CharacterValue::append(std::size_t n, char32_t c) { impl().append(n, c); }

CharacterValue &CharacterValue::replace(
    std::size_t pos, std::size_t len, const CharacterValue &other) {
  impl().replace(pos, len, other.impl());
  return *this;
}

CharacterValue CharacterValue::substr(std::size_t pos) const {
  return FromImpl(impl().substr(pos));
}

CharacterValue CharacterValue::substr(std::size_t pos, std::size_t len) const {
  return FromImpl(impl().substr(pos, len));
}

void CharacterValue::reserve(std::size_t n) { impl().reserve(n); }

char32_t CharacterValue::operator[](std::size_t i) const {
  return impl().operator[](i);
}

CharacterValue CharacterValue::operator+(const CharacterValue &y) const {
  return FromImpl(impl() + y.impl());
}

CharacterValue &CharacterValue::operator+=(const CharacterValue &y) {
  impl() += y.impl();
  return *this;
}

CharacterValue &CharacterValue::operator+=(char c) {
  impl() += c;
  return *this;
}

std::size_t CharacterValue::find(const CharacterValue &pattern) const {
  return impl().find(pattern.impl());
}

std::size_t CharacterValue::rfind(const CharacterValue &pattern) const {
  return impl().rfind(pattern.impl());
}

std::size_t CharacterValue::find_first_of(const CharacterValue &set) const {
  return impl().find_first_of(set.impl());
}

std::size_t CharacterValue::find_last_of(const CharacterValue &set) const {
  return impl().find_last_of(set.impl());
}

std::size_t CharacterValue::find_first_not_of(char32_t c) const {
  return impl().find_first_not_of(c);
}

std::size_t CharacterValue::find_last_not_of(char32_t c) const {
  return impl().find_last_not_of(c);
}

std::size_t CharacterValue::find_first_not_of(const CharacterValue &set) const {
  return impl().find_first_not_of(set.impl());
}

std::size_t CharacterValue::find_last_not_of(const CharacterValue &set) const {
  return impl().find_last_not_of(set.impl());
}

void *CharacterValue::data() { return impl().data(); }
const void *CharacterValue::data() const { return impl().data(); }

void CharacterValue::StoreRawBytes(
    void *dst, size_t size, bool *changed) const {
  impl().StoreRawBytes(dst, size, changed);
}

CharacterValue CharacterValue::FromImpl(const CharacterValueImpl &y) {
  CharacterValue result;
  result.impl() = y;
  return result;
}

CharacterValue CharacterValue::FromImpl(CharacterValueImpl &&y) {
  CharacterValue result;
  result.impl() = std::move(y);
  return result;
}

} // namespace Fortran::evaluate::value
