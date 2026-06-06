// RUN: rm -rf %t
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -fapinotes -fapinotes-modules -fmodules-cache-path=%t -I %S/Inputs/yaml-reader-errors/ -fsyntax-only %s > %t.err 2>&1
// RUN: FileCheck %S/Inputs/yaml-reader-errors/UIKit.apinotes < %t.err

@import UIKit;
