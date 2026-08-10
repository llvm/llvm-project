// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}

// RUN: %clang -print-resource-dir | tr -d '\n' > %t.resource_dir
// RUN: env EXPECTED_RESOURCE_DIR="%{readfile:%t.resource_dir}" && \
// RUN:  mkdir -p %t && rm -rf %t/resource-dir && \
// RUN:  cp -R %{readfile:%t.resource_dir} %t/resource-dir
// RUN: cd %t && %clang -cc1 -x objective-c -fmodules -fmodule-format=obj \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t.mcp \
// RUN:   -fbuiltin-headers-in-system-modules \
// RUN:   -resource-dir resource-dir \
// RUN:   -internal-isystem resource-dir/include \
// RUN:   -emit-module %S/Inputs/builtin-headers/module.modulemap \
// RUN:   -fmodule-name=ModuleWithBuiltinHeader -o %t.pcm
