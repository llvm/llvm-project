//===------------------------- incomplete_type.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// http://mentorembedded.github.io/cxx-abi/abi.html#rtti-layout

// Two abi::__pbase_type_info objects can always be compared for equality
// (i.e. of the types represented) or ordering by comparison of their name
// NTBS addresses. In addition, unless either or both have either of the
// incomplete flags set, equality can be tested by comparing the type_info
// addresses.

// RUN: %cxx %flags %compile_flags -c %s -o %t.one.o
// RUN: %cxx %flags %compile_flags -c %s -o %t.two.o -DTU_ONE
// RUN: %cxx %flags %link_flags -o %t.exe %t.one.o %t.two.o
// RUN: %t.exe

#include <stdio.h>
#include <cassert>
#include <typeinfo>

struct NeverDefined;
void ThrowNeverDefinedMP();

struct IncompleteAtThrow;
void ThrowIncompleteMP();
void ThrowIncompletePP();
void ThrowIncompletePMP();
std::type_info const& ReturnTypeInfoIncompleteMP();
std::type_info const& ReturnTypeInfoIncompletePP();

struct CompleteAtThrow;
void ThrowCompleteMP();
void ThrowCompletePP();
void ThrowCompletePMP();
std::type_info const& ReturnTypeInfoCompleteMP();
std::type_info const& ReturnTypeInfoCompletePP();

void ThrowNullptr();

#ifndef TU_ONE

void ThrowNeverDefinedMP() { throw (int NeverDefined::*)nullptr; }

void ThrowIncompleteMP() { throw (int IncompleteAtThrow::*)nullptr; }
void ThrowIncompletePP() { throw (IncompleteAtThrow**)nullptr; }
void ThrowIncompletePMP() { throw (int IncompleteAtThrow::**)nullptr; }

std::type_info const& ReturnTypeInfoIncompleteMP() { return typeid(int IncompleteAtThrow::*); }
std::type_info const& ReturnTypeInfoIncompletePP() { return typeid(IncompleteAtThrow**); }

struct CompleteAtThrow {};
void ThrowCompleteMP() { throw (int CompleteAtThrow::*)nullptr; }
void ThrowCompletePP() { throw (CompleteAtThrow**)nullptr; }
void ThrowCompletePMP() { throw (int CompleteAtThrow::**)nullptr; }

std::type_info const& ReturnTypeInfoCompleteMP() { return typeid(int CompleteAtThrow::*); }
std::type_info const& ReturnTypeInfoCompletePP() { return typeid(CompleteAtThrow**); }

void ThrowNullptr() { throw nullptr; }

#else

struct IncompleteAtThrow {};

int main() {
  try {
    ThrowNeverDefinedMP();
    assert(false);
  } catch (int IncompleteAtThrow::*) {
    assert(false);
  } catch (int CompleteAtThrow::*) {
    assert(false);
  } catch (int NeverDefined::*) {}

  assert(ReturnTypeInfoIncompleteMP() != typeid(int IncompleteAtThrow::*));
  try {
    ThrowIncompleteMP();
    assert(false);
  } catch (CompleteAtThrow**) {
    assert(false);
  } catch (int CompleteAtThrow::*) {
    assert(false);
  } catch (IncompleteAtThrow**) {
    assert(false);
  } catch (int IncompleteAtThrow::*) {}

  assert(ReturnTypeInfoIncompletePP() != typeid(IncompleteAtThrow**));
  try {
    ThrowIncompletePP();
    assert(false);
  } catch (int IncompleteAtThrow::*) {
    assert(false);
  } catch (IncompleteAtThrow**) {}

  try {
    ThrowIncompletePMP();
    assert(false);
  } catch (int IncompleteAtThrow::*) {
    assert(false);
  } catch (IncompleteAtThrow**) {
    assert(false);
  } catch (int IncompleteAtThrow::**) {}

  assert(ReturnTypeInfoCompleteMP() != typeid(int CompleteAtThrow::*));
  try {
    ThrowCompleteMP();
    assert(false);
  } catch (IncompleteAtThrow**) {
    assert(false);
  } catch (int IncompleteAtThrow::*) {
    assert(false);
  } catch (CompleteAtThrow**) {
    assert(false);
  } catch (int CompleteAtThrow::*) {}

  assert(ReturnTypeInfoCompletePP() != typeid(CompleteAtThrow**));
  try {
    ThrowCompletePP();
    assert(false);
  } catch (IncompleteAtThrow**) {
    assert(false);
  } catch (int IncompleteAtThrow::*) {
    assert(false);
  } catch (int CompleteAtThrow::*) {
    assert(false);
  } catch (CompleteAtThrow**) {}

  try {
    ThrowCompletePMP();
    assert(false);
  } catch (IncompleteAtThrow**) {
    assert(false);
  } catch (int IncompleteAtThrow::*) {
    assert(false);
  } catch (int CompleteAtThrow::*) {
    assert(false);
  } catch (CompleteAtThrow**) {
    assert(false);
  } catch (int CompleteAtThrow::**) {}

#if __cplusplus >= 201103L
  // Catch nullptr as complete type
  try {
    ThrowNullptr();
  } catch (int IncompleteAtThrow::*) {}

  // Catch nullptr as an incomplete type
  try {
    ThrowNullptr();
  } catch (int CompleteAtThrow::*) {}
  // Catch nullptr as a type that is never complete.
  try {
    ThrowNullptr();
  } catch (int NeverDefined::*) {}
#endif
}
#endif
