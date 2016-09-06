//===-- scudo_new_delete.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Interceptors for operators new and delete.
///
//===----------------------------------------------------------------------===//

#include "scudo_allocator.h"

#include "interception/interception.h"

#include <cstddef>

using namespace __scudo;

#define CXX_OPERATOR_ATTRIBUTE INTERCEPTOR_ATTRIBUTE

// Fake std::nothrow_t to avoid including <new>.
namespace std {
struct nothrow_t {};
} // namespace std

CXX_OPERATOR_ATTRIBUTE
void *operator new(size_t size) {
  return scudoMalloc(size, FromNew);
}
CXX_OPERATOR_ATTRIBUTE
void *operator new[](size_t size) {
  return scudoMalloc(size, FromNewArray);
}
CXX_OPERATOR_ATTRIBUTE
void *operator new(size_t size, std::nothrow_t const&) {
  return scudoMalloc(size, FromNew);
}
CXX_OPERATOR_ATTRIBUTE
void *operator new[](size_t size, std::nothrow_t const&) {
  return scudoMalloc(size, FromNewArray);
}

CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr) NOEXCEPT {
  return scudoFree(ptr, FromNew);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr) NOEXCEPT {
  return scudoFree(ptr, FromNewArray);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, std::nothrow_t const&) NOEXCEPT {
  return scudoFree(ptr, FromNew);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, std::nothrow_t const&) NOEXCEPT {
  return scudoFree(ptr, FromNewArray);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, size_t size) NOEXCEPT {
  scudoSizedFree(ptr, size, FromNew);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, size_t size) NOEXCEPT {
  scudoSizedFree(ptr, size, FromNewArray);
}
