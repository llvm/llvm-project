// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodule-cache-path %t -fmodules -F %S/Inputs -include Module/Module.h %s -emit-llvm -o - | FileCheck %s

// CHECK: call i8* @getModuleVersion
const char* getVer(void) {
  return getModuleVersion();
}
