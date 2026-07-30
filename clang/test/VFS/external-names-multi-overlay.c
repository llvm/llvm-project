// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@NAME_DIR@%{/t:regex_replacement}/A@g" -e "s@EXTERNAL_DIR@%{/t:regex_replacement}/B@g" -e "s@REDIRECT_WITH@fallthrough@g" %t/vfs/base.yaml > %t/vfs/a-b-ft.yaml
// RUN: sed -e "s@NAME_DIR@%{/t:regex_replacement}/A@g" -e "s@EXTERNAL_DIR@%{/t:regex_replacement}/B@g" -e "s@REDIRECT_WITH@fallback@g" %t/vfs/base.yaml > %t/vfs/a-b-fb.yaml

// Check that the external name is given when multiple overlays are provided

// RUN: %clang_cc1 -Werror -I %t/A -ivfsoverlay %t/vfs/a-b-ft.yaml -ivfsoverlay %t/vfs/empty.yaml -E -C %t/main.c 2>&1 | FileCheck --check-prefix=FROM_B %s
// RUN: %clang_cc1 -Werror -I %t/A -ivfsoverlay %t/vfs/a-b-fb.yaml -ivfsoverlay %t/vfs/empty.yaml -E -C %t/main.c 2>&1 | FileCheck --check-prefix=FROM_B %s
// RUN: %clang_cc1 -Werror -I %t/B -ivfsoverlay %t/vfs/a-b-ft.yaml -ivfsoverlay %t/vfs/empty.yaml -E -C %t/main.c 2>&1 | FileCheck --check-prefix=FROM_B %s
// RUN: %clang_cc1 -Werror -I %t/B -ivfsoverlay %t/vfs/a-b-fb.yaml -ivfsoverlay %t/vfs/empty.yaml -E -C %t/main.c 2>&1 | FileCheck --check-prefix=FROM_B %s
// FROM_B: # 1 "{{.*(/|\\\\)B(/|\\\\)}}Header.h"
// FROM_B: // Header.h in B

//--- main.c
#include "Header.h"

//--- B/Header.h
// Header.h in B

//--- vfs/base.yaml
{
  'version': 0,
  'redirecting-with': 'REDIRECT_WITH',
  'roots': [
    { 'name': 'NAME_DIR',
      'type': 'directory-remap',
      'external-contents': 'EXTERNAL_DIR'
    }
  ]
}

//--- vfs/empty.yaml
{
  'version': 0,
  'roots': []
}
