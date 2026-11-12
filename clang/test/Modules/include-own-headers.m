// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodule-name=Module -fimplicit-module-maps -fmodules-cache-path=%t -Werror=non-modular-include-in-framework-module -F%S/Inputs -I%S -fsyntax-only %s
#include "Module/Module.h"
#include "Inputs/non-module.h"
