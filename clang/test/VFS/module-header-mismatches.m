// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s;TEST_DIR;%/t;g" %t/sed-overlay.yaml > %t/overlay.yaml

// These tests first build with an overlay such that the header is resolved
// to %t/other/Mismatch.h. They then build again with the header resolved
// to the one in their directory.
//
// This should cause a rebuild if the contents is different (and thus multiple
// PCMs), but this currently isn't the case. We should at least not error,
// since this does happen in real projects (with a different copy of the same
// file).

// RUN: %clang_cc1 -Werror -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/hf-mcp -ivfsoverlay %t/overlay.yaml -F %t/header-frameworks -fsyntax-only -verify %t/use.m
// RUN: %clang_cc1 -Werror -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/hf-mcp -F %t/header-frameworks -fsyntax-only -verify %t/use.m
// RUN: find %t/hf-mcp -name "Mismatch-*.pcm" | count 1

// RUN: %clang_cc1 -Werror -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/df-mcp -ivfsoverlay %t/overlay.yaml -F %t/dir-frameworks -fsyntax-only -verify %t/use.m
// RUN: %clang_cc1 -Werror -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/hf-mcp -F %t/dir-frameworks -fsyntax-only -verify %t/use.m
// RUN: find %t/df-mcp -name "Mismatch-*.pcm" | count 1

// RUN: %clang_cc1 -Werror -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/nf-mcp -ivfsoverlay %t/overlay.yaml -F %t/norm-frameworks -fsyntax-only -verify %t/use.m
// RUN: %clang_cc1 -Werror -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/nf-mcp -F %t/norm-frameworks -fsyntax-only -verify %t/use.m
// RUN: find %t/nf-mcp -name "Mismatch-*.pcm" | count 1

// RUN: %clang_cc1 -Werror -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/m-mcp -ivfsoverlay %t/overlay.yaml -I %t/mod -fsyntax-only -verify %t/use.m
// RUN: %clang_cc1 -Werror -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/m-mcp -I %t/mod -fsyntax-only -verify %t/use.m
// RUN: find %t/m-mcp -name "Mismatch-*.pcm" | count 1

//--- use.m
// expected-no-diagnostics
@import Mismatch;

//--- header-frameworks/Mismatch.framework/Modules/module.modulemap
framework module Mismatch {
  umbrella header "Mismatch.h"
}
//--- header-frameworks/Mismatch.framework/Headers/Mismatch.h

//--- dir-frameworks/Mismatch.framework/Modules/module.modulemap
framework module Mismatch {
  umbrella "someheaders"
}
//--- dir-frameworks/Mismatch.framework/someheaders/Mismatch.h

//--- norm-frameworks/Mismatch.framework/Modules/module.modulemap
framework module Mismatch {
  header "Mismatch.h"
}
//--- norm-frameworks/Mismatch.framework/Headers/Mismatch.h

//--- mod/module.modulemap
module Mismatch {
  umbrella header "Mismatch.h"
}
//--- mod/Mismatch.h

//--- other/Mismatch.h

//--- sed-overlay.yaml
{
  'version': 0,
  'roots': [
    { 'name': 'TEST_DIR', 'type': 'directory',
      'contents': [
        { 'name': 'header-frameworks/Mismatch.framework/Headers/Mismatch.h',
          'type': 'file',
          'external-contents': 'TEST_DIR/other/Mismatch.h'
        },
        { 'name': 'dir-frameworks/Mismatch.framework/someheaders',
          'type': 'directory',
          'external-contents': 'TEST_DIR/others'
        },
        { 'name': 'norm-frameworks/Mismatch.framework/Headers/Mismatch.h',
          'type': 'file',
          'external-contents': 'TEST_DIR/other/Mismatch.h'
        },
        { 'name': 'mod/Mismatch.h',
          'type': 'file',
          'external-contents': 'TEST_DIR/other/Mismatch.h'
        }
      ]
    }
  ]
}

