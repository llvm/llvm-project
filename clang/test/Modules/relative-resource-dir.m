// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// REQUIRES: shell

// RUN: EXPECTED_RESOURCE_DIR=`%clang -print-resource-dir` && \
// RUN:  mkdir -p %t && rm -rf %t/resource-dir && \
// RUN:  cp -R $EXPECTED_RESOURCE_DIR %t/resource-dir
// RUN: cd %t && %clang -cc1 -x objective-c -fmodules -fmodule-format=obj \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t.mcp \
// RUN:   -fbuiltin-headers-in-system-modules \
// RUN:   -resource-dir resource-dir \
// RUN:   -emit-module %S/Inputs/builtin-headers/module.modulemap \
// RUN:   -fmodule-name=ModuleWithBuiltinHeader -o %t.pcm
