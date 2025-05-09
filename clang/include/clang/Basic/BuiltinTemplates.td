//===--- BuiltinTemplates.td - Clang builtin template aliases ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class TemplateArg<string name> {
  string Name = name;
}

class Template<list<TemplateArg> args, string name> : TemplateArg<name> {
  list<TemplateArg> Args = args;
}

class Class<string name, bit is_variadic = 0> : TemplateArg<name> {
  bit IsVariadic = is_variadic;
}

class NTTP<string type_name, string name, bit is_variadic = 0> : TemplateArg<name> {
  string TypeName = type_name;
  bit IsVariadic = is_variadic;
}

class BuiltinNTTP<string type_name> : TemplateArg<""> {
  string TypeName = type_name;
}

def SizeT : BuiltinNTTP<"size_t"> {}

class BuiltinTemplate<list<TemplateArg> template_head> {
  list<TemplateArg> TemplateHead = template_head;
}

// template <template <class T, T... Ints> IntSeq, class T, T N>
def __make_integer_seq : BuiltinTemplate<
  [Template<[Class<"T">, NTTP<"T", "Ints", /*is_variadic=*/1>], "IntSeq">, Class<"T">, NTTP<"T", "N">]>;

// template <size_t, class... T>
def __type_pack_element : BuiltinTemplate<
  [SizeT, Class<"T", /*is_variadic=*/1>]>;

// template <template <class... Args> BaseTemplate,
//           template <class TypeMember> HasTypeMember,
//           class HasNoTypeMember
//           class... Ts>
def __builtin_common_type : BuiltinTemplate<
  [Template<[Class<"Args", /*is_variadic=*/1>], "BaseTemplate">,
   Template<[Class<"TypeMember">], "HasTypeMember">,
   Class<"HasNoTypeMember">,
   Class<"Ts", /*is_variadic=*/1>]>;
