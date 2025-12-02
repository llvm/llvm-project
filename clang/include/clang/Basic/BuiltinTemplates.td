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

class Template<list<TemplateArg> args, string name = ""> : TemplateArg<name> {
  list<TemplateArg> Args = args;
}

class Class<string name = "", bit is_variadic = 0> : TemplateArg<name> {
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
def Uint32T: BuiltinNTTP<"uint32_t"> {}

class BuiltinTemplate<list<TemplateArg> template_head> {
  list<TemplateArg> TemplateHead = template_head;
}

class CPlusPlusBuiltinTemplate<list<TemplateArg> template_head> : BuiltinTemplate<template_head>;

class HLSLBuiltinTemplate<list<TemplateArg> template_head> : BuiltinTemplate<template_head>;

// template <template <class T, T... Ints> IntSeq, class T, T N>
def __make_integer_seq : CPlusPlusBuiltinTemplate<
  [Template<[Class<"T">, NTTP<"T", "Ints", /*is_variadic=*/1>], "IntSeq">, Class<"T">, NTTP<"T", "N">]>;

// template <size_t, class... T>
def __type_pack_element : CPlusPlusBuiltinTemplate<
  [SizeT, Class<"T", /*is_variadic=*/1>]>;

// template <template <class... Args> BaseTemplate,
//           template <class TypeMember> HasTypeMember,
//           class HasNoTypeMember
//           class... Ts>
def __builtin_common_type : CPlusPlusBuiltinTemplate<
  [Template<[Class<"Args", /*is_variadic=*/1>], "BaseTemplate">,
   Template<[Class<"TypeMember">], "HasTypeMember">,
   Class<"HasNoTypeMember">,
   Class<"Ts", /*is_variadic=*/1>]>;

// template <template <class,"
//                     class,"
//                     template <class> class,"
//                     template <class> class> class BasicCommonReferenceT,"
//           template <class... Args> class CommonTypeT,"
//           template <class> class HasTypeMember,"
//           class HasNoTypeMember,"
//           class... Ts>"
def __builtin_common_reference : CPlusPlusBuiltinTemplate<
            [Template<[Class<>,
                       Class<>,
                       Template<[Class<>]>,
                       Template<[Class<>]>], "BasicCommonReferenceT">,
             Template<[Class<"Args", /*is_variadic=*/1>], "CommonTypeT">,
             Template<[Class<>], "HasTypeMember">,
             Class<"HasNoTypeMember">,
             Class<"Ts", /*is_variadic=*/1>]>;

foreach Ref = ["", "lvalue", "rvalue"] in {
  foreach Const = ["", "const"] in {
    foreach Volatile = ["", "volatile"] in {
      def __clang_internal_xref_#Ref#Const#Volatile : CPlusPlusBuiltinTemplate<[Class<>]>;
    }
  }
}

// template <uint32_t Opcode,
//           uint32_t Size,
//           uint32_t Alignment,
//           typename ...Operands>
def __hlsl_spirv_type : HLSLBuiltinTemplate<
[Uint32T, Uint32T, Uint32T, Class<"Operands", /*is_variadic=*/1>]>;

// template <class ...Args>
def __builtin_dedup_pack
    : CPlusPlusBuiltinTemplate<[Class<"Args", /*is_variadic=*/1>]>;
