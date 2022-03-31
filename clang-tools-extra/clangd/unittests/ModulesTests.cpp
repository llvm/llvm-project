//===-- ModulesTests.cpp  ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFS.h"
#include "TestTU.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace clang {
namespace clangd {
namespace {

TEST(Modules, TextualIncludeInPreamble) {
  TestTU TU = TestTU::withCode(R"cpp(
    #include "Textual.h"

    void foo() {}
)cpp");
  TU.ExtraArgs.push_back("-fmodule-name=M");
  TU.ExtraArgs.push_back("-fmodule-map-file=" + testPath("m.modulemap"));
  TU.AdditionalFiles["Textual.h"] = "void foo();";
  TU.AdditionalFiles["m.modulemap"] = R"modulemap(
    module M {
      module Textual {
        textual header "Textual.h"
      }
    }
)modulemap";
  // Test that we do not crash.
  TU.index();
}

// Verify that visibility of AST nodes belonging to modules, but loaded from
// preamble PCH, is restored.
TEST(Modules, PreambleBuildVisibility) {
  TestTU TU = TestTU::withCode(R"cpp(
    #include "module.h"

    foo x;
)cpp");
  TU.OverlayRealFileSystemForModules = true;
  TU.ExtraArgs.push_back("-fmodules");
  TU.ExtraArgs.push_back("-fmodules-strict-decluse");
  TU.ExtraArgs.push_back("-Xclang");
  TU.ExtraArgs.push_back("-fmodules-local-submodule-visibility");
  TU.ExtraArgs.push_back("-fmodule-map-file=" + testPath("m.modulemap"));
  TU.AdditionalFiles["module.h"] = R"cpp(
    typedef int foo;
)cpp";
  TU.AdditionalFiles["m.modulemap"] = R"modulemap(
    module M {
      header "module.h"
    }
)modulemap";
  EXPECT_TRUE(TU.build().getDiagnostics()->empty());
}

TEST(Modules, Diagnostic) {
  // Produce a diagnostic while building an implicit module. Use
  // -fmodules-strict-decluse, but any non-silenced diagnostic will do.
  TestTU TU = TestTU::withCode(R"cpp(
    /*error-ok*/
    #include "modular.h"

    void bar() {}
)cpp");
  TU.OverlayRealFileSystemForModules = true;
  TU.ExtraArgs.push_back("-fmodule-map-file=" + testPath("m.modulemap"));
  TU.ExtraArgs.push_back("-fmodules");
  TU.ExtraArgs.push_back("-fimplicit-modules");
  TU.ExtraArgs.push_back("-fmodules-strict-decluse");
  TU.AdditionalFiles["modular.h"] = R"cpp(
    #include "non-modular.h"
  )cpp";
  TU.AdditionalFiles["non-modular.h"] = "";
  TU.AdditionalFiles["m.modulemap"] = R"modulemap(
    module M {
      header "modular.h"
    }
)modulemap";

  // Test that we do not crash.
  TU.build();
}

// Unknown module formats are a fatal failure for clang. Ensure we don't crash.
TEST(Modules, UnknownFormat) {
  TestTU TU = TestTU::withCode(R"(#include "modular.h")");
  TU.OverlayRealFileSystemForModules = true;
  TU.ExtraArgs.push_back("-Xclang");
  TU.ExtraArgs.push_back("-fmodule-format=obj");
  TU.ExtraArgs.push_back("-fmodule-map-file=" + testPath("m.modulemap"));
  TU.ExtraArgs.push_back("-fmodules");
  TU.ExtraArgs.push_back("-fimplicit-modules");
  TU.AdditionalFiles["modular.h"] = "";
  TU.AdditionalFiles["m.modulemap"] = R"modulemap(
    module M {
      header "modular.h"
    })modulemap";

  // Test that we do not crash.
  TU.build();
}
} // namespace
} // namespace clangd
} // namespace clang
