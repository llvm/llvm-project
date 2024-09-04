//===--- BuiltinTemplates.td - Clang builtin template aliases ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class BuiltinTemplate<string prototype> {
  string Prototype = prototype;
}

def __make_integer_seq : BuiltinTemplate<
  "template <template <class T, T... Ints> class IntSeq, class T, T N>">;

def __type_pack_element : BuiltinTemplate<
  "template <size_t, class... T>">;

def __builtin_common_type : BuiltinTemplate<
  "template <template <class... Args> class BaseTemplate,"
  "          template <class TypeMember> class HasTypeMember,"
  "          class HasNoTypeMember,"
  "          class... Ts>">;
