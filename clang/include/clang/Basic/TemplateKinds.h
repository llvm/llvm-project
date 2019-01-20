//===--- TemplateKinds.h - Enum values for C++ Template Kinds ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the clang::TemplateNameKind enum.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_BASIC_TEMPLATEKINDS_H
#define LLVM_CLANG_BASIC_TEMPLATEKINDS_H

namespace clang {

/// Specifies the kind of template name that an identifier refers to.
/// Be careful when changing this: this enumeration is used in diagnostics.
enum TemplateNameKind {
  /// The name does not refer to a template.
  TNK_Non_template = 0,
  /// The name refers to a function template or a set of overloaded
  /// functions that includes at least one function template.
  TNK_Function_template,
  /// The name refers to a template whose specialization produces a
  /// type. The template itself could be a class template, template
  /// template parameter, or template alias.
  TNK_Type_template,
  /// The name refers to a variable template whose specialization produces a
  /// variable.
  TNK_Var_template,
  /// The name refers to a dependent template name:
  /// \code
  /// template<typename MetaFun, typename T1, typename T2> struct apply2 {
  ///   typedef typename MetaFun::template apply<T1, T2>::type type;
  /// };
  /// \endcode
  ///
  /// Here, "apply" is a dependent template name within the typename
  /// specifier in the typedef. "apply" is a nested template, and
  /// whether the template name is assumed to refer to a type template or a
  /// function template depends on the context in which the template
  /// name occurs.
  TNK_Dependent_template_name
};

}
#endif


