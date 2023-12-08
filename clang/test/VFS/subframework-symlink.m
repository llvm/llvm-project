// FIXME: PR44221
// UNSUPPORTED: system-windows

// Test that when a subframework is a symlink to another framework, we don't
// add it as a submodule to the enclosing framework. We also need to make clang
// to infer module for the enclosing framework. For this we don't have
// a module map for the framework itself but have it in a parent directory.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'framework module * {}' > %t/module.modulemap
// RUN: mkdir -p %t/WithSubframework.framework/Headers
// RUN: echo '#include <Foo/Foo.h>' > %t/WithSubframework.framework/Headers/WithSubframework.h
// RUN: cp -R %S/Inputs/Foo.framework %t
// RUN: mkdir -p %t/WithSubframework.framework/Frameworks
// RUN: ln -s %t/Foo.framework %t/WithSubframework.framework/Frameworks
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache1 -F %t -fsyntax-only %s

// Adding VFS overlay shouldn't change this behavior.
//
// RUN: sed -e "s@INPUT_DIR@/InvalidPath@g" -e "s@OUT_DIR@/InvalidPath@g" %S/Inputs/vfsoverlay.yaml > %t/overlay.yaml
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache2 -F %t -fsyntax-only %s -ivfsoverlay %t/overlay.yaml

#import <WithSubframework/WithSubframework.h>
