// RUN: rm -rf %t && mkdir -p %t
// RUN: echo "void funcA(void);" >> %t/a.h

// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/mcp -I %S/Inputs/MissingVFS %s -fsyntax-only -ivfsoverlay %t/vfs.yaml 2>&1 | FileCheck %s -check-prefix=ERROR
// ERROR: virtual filesystem overlay file '{{.*}}' not found
// RUN: find %t/mcp -name "A-*.pcm" | count 1

// RUN: sed -e "s@INPUT_DIR@%{/S:regex_replacement}/Inputs@g" -e "s@OUT_DIR@%{/t:regex_replacement}@g" %S/Inputs/MissingVFS/vfsoverlay.yaml > %t/vfs.yaml
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/mcp -I %S/Inputs/MissingVFS %s -fsyntax-only -ivfsoverlay %t/vfs.yaml
// RUN: find %t/mcp -name "A-*.pcm" | count 1

@import A;
void test(void) {
  funcA();
}
