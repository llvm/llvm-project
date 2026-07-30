// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs -include Module/Module.h %s -emit-llvm -o - | FileCheck %s

// CHECK: call {{.*}}ptr @getModuleVersion
const char* getVer(void) {
  return getModuleVersion();
}
