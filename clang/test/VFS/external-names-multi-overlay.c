// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@NAME_DIR@%{/t:regex_replacement}/A@g" -e "s@EXTERNAL_DIR@%{/t:regex_replacement}/B@g" %t/vfs/base.yaml > %t/vfs/a-b.yaml
// RUN: sed -e "s@NAME_DIR@%{/t:regex_replacement}/C@g" -e "s@EXTERNAL_DIR@%{/t:regex_replacement}/D@g" %t/vfs/base.yaml > %t/vfs/c-d.yaml

// Check that the external name is given when multiple overlays are provided

// RUN: %clang_cc1 -Werror -I %t/A -ivfsoverlay %t/vfs/c-d.yaml -ivfsoverlay %t/vfs/a-b.yaml -fsyntax-only -E -C %t/main.c 2>&1 | FileCheck --check-prefix=FROM_B %s
// FROM_B: # 1 "{{.*(/|\\\\)B(/|\\\\)}}Header.h"
// FROM_B: // Header.h in B

// RUN: %clang_cc1 -Werror -I %t/B -ivfsoverlay %t/vfs/c-d.yaml -ivfsoverlay %t/vfs/a-b.yaml -fsyntax-only -E -C %t/main.c 2>&1 | FileCheck --check-prefix=FROM_B %s

// RUN: %clang_cc1 -Werror -I %t/C -ivfsoverlay %t/vfs/c-d.yaml -ivfsoverlay %t/vfs/a-b.yaml -fsyntax-only -E -C %t/main.c 2>&1 | FileCheck --check-prefix=FROM_D %s
// FROM_D: # 1 "{{.*(/|\\\\)D(/|\\\\)}}Header.h"
// FROM_D: // Header.h in D

// RUN: %clang_cc1 -Werror -I %t/C -ivfsoverlay %t/vfs/c-d.yaml -ivfsoverlay %t/vfs/a-b.yaml -fsyntax-only -E -C %t/main.c 2>&1 | FileCheck --check-prefix=FROM_D %s

//--- main.c
#include "Header.h"

//--- B/Header.h
// Header.h in B

//--- D/Header.h
// Header.h in D

//--- vfs/base.yaml
{
  'version': 0,
  'redirecting-with': 'fallthrough',
  'roots': [
    { 'name': 'NAME_DIR',
      'type': 'directory-remap',
      'external-contents': 'EXTERNAL_DIR'
    }
  ]
}
