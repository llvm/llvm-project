// Test the module map path is consistent between clang invocations when using VFS overlays.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Pre-populate the module cache with the modules that don't use VFS overlays.
// RUN: %clang_cc1 -fsyntax-only -F%t/Frameworks -I%t/include %t/prepopulate_module_cache.m \
// RUN:     -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// Execute a compilation with VFS overlay. .pcm file path looks like <hash1>/ModuleName-<hash2>.pcm.
// <hash1> corresponds to the compilation settings like language options.
// <hash2> corresponds to the module map path. So if any of those change, we should use a different module.
// But for VFS overlay we make an exception that it's not a part of <hash1> to reduce the number of built .pcm files.
// Test that paths in overlays don't leak into <hash2> and don't cause using 2 .pcm files for the same module.
// DEFINE: %{command} = %clang_cc1 -fsyntax-only -verify -F%t/Frameworks -I%t/include %t/test.m \
// DEFINE:    -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache
// RUN: sed -e "s@TMP_DIR@%{/t:regex_replacement}@g" -e "s@USE_EXTERNAL_NAMES_OPTION@@g" %t/overlay.yaml.template > %t/external-names-default.yaml
// RUN: %{command} -ivfsoverlay %t/external-names-default.yaml

// RUN: sed -e "s@TMP_DIR@%{/t:regex_replacement}@g" -e "s@USE_EXTERNAL_NAMES_OPTION@'use-external-names': true,@g" %t/overlay.yaml.template > %t/external-names-true.yaml
// RUN: %{command} -ivfsoverlay %t/external-names-true.yaml

// RUN: sed -e "s@TMP_DIR@%{/t:regex_replacement}@g" -e "s@USE_EXTERNAL_NAMES_OPTION@'use-external-names': false,@g" %t/overlay.yaml.template > %t/external-names-false.yaml
// RUN: %{command} -ivfsoverlay %t/external-names-false.yaml

//--- prepopulate_module_cache.m
#import <Redirecting/Redirecting.h>

//--- test.m
// At first import multi-path modules directly, so clang decides which .pcm file they should belong to.
#import <MultiPath/MultiPath.h>
#import <MultiPathHeader.h>

// Then import a module from the module cache and all its transitive dependencies.
// Make sure the .pcm files loaded directly are the same as 'Redirecting' is referencing.
#import <Redirecting/Redirecting.h>
// expected-no-diagnostics


//--- Frameworks/MultiPath.framework/Headers/MultiPath.h
void multiPathFramework(void);

//--- Frameworks/MultiPath.framework/Modules/module.modulemap
framework module MultiPath {
    header "MultiPath.h"
    export *
}


//--- include/MultiPathHeader.h
void multiPathHeader(void);

//--- include/module.modulemap
module MultiPathHeader {
    header "MultiPathHeader.h"
    export *
}


//--- Frameworks/Redirecting.framework/Headers/Redirecting.h
#import <MultiPath/MultiPath.h>
#import <MultiPathHeader.h>

//--- Frameworks/Redirecting.framework/Modules/module.modulemap
framework module Redirecting {
    header "Redirecting.h"
    export *
}


//--- BuildTemporaries/MultiPath.h
void multiPathFramework(void);
//--- BuildTemporaries/framework.modulemap
framework module MultiPath {
    header "MultiPath.h"
    export *
}
//--- BuildTemporaries/header.h
void multiPathHeader(void);
//--- BuildTemporaries/include.modulemap
module MultiPathHeader {
    header "MultiPathHeader.h"
    export *
}

//--- overlay.yaml.template
{
  'version': 0,
  USE_EXTERNAL_NAMES_OPTION
  'roots': [
    { 'name': 'TMP_DIR/Frameworks/MultiPath.framework/Headers', 'type': 'directory',
      'contents': [
        { 'name': 'MultiPath.h', 'type': 'file',
          'external-contents': 'TMP_DIR/BuildTemporaries/MultiPath.h'}
    ]},
    { 'name': 'TMP_DIR/Frameworks/MultiPath.framework/Modules', 'type': 'directory',
      'contents': [
        { 'name': 'module.modulemap', 'type': 'file',
          'external-contents': 'TMP_DIR/BuildTemporaries/framework.modulemap'}
    ]},
    { 'name': 'TMP_DIR/include', 'type': 'directory',
      'contents': [
        { 'name': 'MultiPathHeader.h', 'type': 'file',
          'external-contents': 'TMP_DIR/BuildTemporaries/header.h'},
        { 'name': 'module.modulemap', 'type': 'file',
          'external-contents': 'TMP_DIR/BuildTemporaries/include.modulemap'}
    ]}
  ]
}

