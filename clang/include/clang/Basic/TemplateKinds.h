//===--- TemplateKinds.h - Enum values for C++ Template Kinds ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the clang::TemplateNameKind enum.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TEMPLATEKINDS_H
#define LLVM_CLANG_TEMPLATEKINDS_H

namespace clang {

/// \brief Specifies the kind of template name that an identifier refers to.
enum TemplateNameKind {
  /// The name does not refer to a template.
  TNK_Non_template = 0,
  /// The name refers to a function template or a set of overloaded
  /// functions that includes at least one function template.
  TNK_Function_template,
  /// The name refers to a template whose specialization produces a
  /// type. The template itself could be a class template, template
  /// template parameter, or C++0x template alias.
  TNK_Type_template,
  /// The name refers to a dependent template name. Whether the
  /// template name is assumed to refer to a type template or a
  /// function template depends on the context in which the template
  /// name occurs.
  TNK_Dependent_template_name
};

}
#endif


