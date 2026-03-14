// RUN: rm -rf %t
// RUN: split-file %s %t

// Test fallback directory remapping, ie. a directory "Base" which is used as
// a fallback if files are missing from "UseFirst"

// RUN: sed -e "s@EXTERNAL_DIR@%{/t:regex_replacement}/Both/Base@g" -e "s@NAME_DIR@%{/t:regex_replacement}/Both/UseFirst@g" %t/vfs/base.yaml > %t/vfs/both.yaml

// RUN: cp -R %t/Both %t/UseFirstOnly
// RUN: rm -rf %t/UseFirstOnly/Base
// RUN: sed -e "s@EXTERNAL_DIR@%{/t:regex_replacement}/UseFirstOnly/Base@g" -e "s@NAME_DIR@%{/t:regex_replacement}/UseFirstOnly/UseFirst@g" %t/vfs/base.yaml > %t/vfs/use-first-only.yaml

// RUN: cp -R %t/Both %t/BaseOnly
// RUN: rm -rf %t/BaseOnly/UseFirst
// RUN: sed -e "s@EXTERNAL_DIR@%{/t:regex_replacement}/BaseOnly/Base@g" -e "s@NAME_DIR@%{/t:regex_replacement}/BaseOnly/UseFirst@g" %t/vfs/base.yaml > %t/vfs/base-only.yaml

// RUN: cp -R %t/Both %t/BFallback
// RUN: rm %t/BFallback/UseFirst/B.h
// RUN: sed -e "s@EXTERNAL_DIR@%{/t:regex_replacement}/BFallback/Base@g" -e "s@NAME_DIR@%{/t:regex_replacement}/BFallback/UseFirst@g" %t/vfs/base.yaml > %t/vfs/b-fallback.yaml

// RUN: cp -R %t/Both %t/CFallback
// RUN: rm %t/CFallback/UseFirst/C.h
// RUN: sed -e "s@EXTERNAL_DIR@%{/t:regex_replacement}/CFallback/Base@g" -e "s@NAME_DIR@%{/t:regex_replacement}/CFallback/UseFirst@g" %t/vfs/base.yaml > %t/vfs/c-fallback.yaml

// Both B.h and C.h are in both folders
// RUN: %clang_cc1 -Werror -I %t/Both/UseFirst -ivfsoverlay %t/vfs/both.yaml -E -C %t/main.c 2>&1 | FileCheck --check-prefix=IN_UF %s

// IN_UF: # 1 "{{.*(/|\\\\)UseFirst(/|\\\\)}}B.h"
// IN_UF-NEXT: // B.h in UseFirst
// IN_UF: # 1 "{{.*(/|\\\\)UseFirst(/|\\\\)}}C.h"
// IN_UF-NEXT: // C.h in UseFirst

// Base missing, so now they are only in UseFirst
// RUN: %clang_cc1 -Werror -I %t/UseFirstOnly/UseFirst -ivfsoverlay %t/vfs/use-first-only.yaml -E -C %t/main.c 2>&1 | FileCheck --check-prefix=IN_UF %s

// UseFirst missing, fallback to Base
// RUN: %clang_cc1 -Werror -I %t/BaseOnly/UseFirst -ivfsoverlay %t/vfs/base-only.yaml -E -C %t/main.c 2>&1 | FileCheck --check-prefix=IN_BASE %s

// IN_BASE: # 1 "{{.*(/|\\\\)Base(/|\\\\)}}B.h"
// IN_BASE-NEXT: // B.h in Base
// IN_BASE: # 1 "{{.*(/|\\\\)Base(/|\\\\)}}C.h"
// IN_BASE-NEXT: // C.h in Base

// B.h missing from UseFirst
// RUN: %clang_cc1 -Werror -I %t/BFallback/UseFirst -ivfsoverlay %t/vfs/b-fallback.yaml -E -C %t/main.c 2>&1 | FileCheck --check-prefix=B_FALLBACK %s

// B_FALLBACK: # 1 "{{.*(/|\\\\)Base(/|\\\\)}}B.h"
// B_FALLBACK-NEXT: // B.h in Base
// B_FALLBACK: # 1 "{{.*(/|\\\\)UseFirst(/|\\\\)}}C.h"
// B_FALLBACK-NEXT: // C.h in UseFirst

// C.h missing from UseFirst
// RUN: %clang_cc1 -Werror -I %t/CFallback/UseFirst -ivfsoverlay %t/vfs/c-fallback.yaml -E -C %t/main.c 2>&1 | FileCheck --check-prefix=C_FALLBACK %s

// C_FALLBACK: # 1 "{{.*(/|\\\\)UseFirst(/|\\\\)}}B.h"
// C_FALLBACK-NEXT: // B.h in UseFirst
// C_FALLBACK: # 1 "{{.*(/|\\\\)Base(/|\\\\)}}C.h"
// C_FALLBACK-NEXT: // C.h in Base

//--- main.c
#include "B.h"

//--- Both/UseFirst/B.h
// B.h in UseFirst
#include "C.h"

//--- Both/UseFirst/C.h
// C.h in UseFirst

//--- Both/Base/B.h
// B.h in Base
#include "C.h"

//--- Both/Base/C.h
// C.h in Base

//--- vfs/base.yaml
{
  'version' : 0,
      'redirecting-with' : 'fallback',
                           'roots' : [
                             {'name' : 'NAME_DIR',
                              'type' : 'directory-remap',
                              'external-contents' : 'EXTERNAL_DIR'}
                           ]
}
