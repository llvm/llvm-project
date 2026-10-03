//===-- DirectivesPlugin.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Example plugin defining compiler directives with the prefix "example"
// (flang/Support/PluginDirectives.h). Loaded with `flang -fc1 -load`, it makes
// flang accept, check and lower
//
//   !DIR$ EXAMPLE CALLBACK([proc,] HANDLER=proc [, PRIORITY=n] [, TAG=str])
//   !DIR$ EXAMPLE WATCH(var [, BY=var])
//   !DIR$ EXAMPLE NOTE([proc-or-var,] TEXT=str)
//   !DIR$ EXAMPLE CONVERGE(var [, var]... [, TOL=real]) [MAX_ITERS(n)]
//
// which may also be spelled with the plugin's own comment sentinel, e.g.
// `!$EXAMPLE NOTE(TEXT="...")`, a comment for other compilers.
//
// Each one but CONVERGE becomes an entry of the `fir.directives` attribute of
// the func.func or fir.global of its subject. CONVERGE applies to the DO or
// DO WHILE loop that follows it: its variables are passed to a marker call
// at the start of the loop body. A pass of the plugin would act on them;
// this one defines no pass.
//
//===----------------------------------------------------------------------===//

#include "flang/Support/PluginDirectives.h"
#include <utility>

using namespace Fortran::common;

namespace {

[[maybe_unused]] const bool registered{[] {
  // Call HANDLER when the subject procedure (by default, the subprogram the
  // directive is in) is called.
  registerPluginDirective(
      {"example", "callback", PluginDirectiveSubject::Procedure,
          {{"handler", PluginDirectiveArgKind::Procedure, /*required=*/true},
              {"priority", PluginDirectiveArgKind::Integer},
              {"tag", PluginDirectiveArgKind::String}}});
  // Track the subject variable, or a COMMON block, possibly through another
  // variable.
  registerPluginDirective({"example", "watch", PluginDirectiveSubject::Variable,
      {{"by", PluginDirectiveArgKind::Variable}}});
  // A remark on a procedure or a variable.
  registerPluginDirective({"example", "note", PluginDirectiveSubject::Any,
      {{"text", PluginDirectiveArgKind::String, /*required=*/true}}});
  // The loop that follows iterates the variables until they converge.
  PluginDirectiveSpec converge{"example", "converge",
      PluginDirectiveSubject::Loop,
      {{"tol", PluginDirectiveArgKind::Real},
          {"max_iters", PluginDirectiveArgKind::Integer}}};
  converge.minPositional = 1;
  registerPluginDirective(std::move(converge));
  // !$example ... is !DIR$ example ...
  registerPluginDirectiveSentinel("example");
  return true;
}()};

} // namespace
