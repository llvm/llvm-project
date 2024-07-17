// RUN: rm -rf %t
// RUN: mkdir -p %t/Underlying
// RUN: mkdir -p %t/Overlay
// RUN: mkdir -p %t/Middle
// RUN: echo '// B.h in Underlying' > %t/Underlying/B.h
// RUN: echo '#ifdef NESTED' >> %t/Underlying/B.h
// RUN: echo '#include "C.h"' >> %t/Underlying/B.h
// RUN: echo '#endif' >> %t/Underlying/B.h
// RUN: echo '// C.h in Underlying' > %t/Underlying/C.h
// RUN: echo '// C.h in Middle' > %t/Middle/C.h
// RUN: echo '// C.h in Overlay' > %t/Overlay/C.h

// 1) Underlying -> Overlay (C.h found, B.h falling back to Underlying)
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}/Overlay@g" -e "s@OUT_DIR@%{/t:regex_replacement}/Underlying@g" %S/Inputs/vfsoverlay-directory.yaml > %t/vfs.yaml
// RUN: %clang_cc1 -Werror -I %t/Underlying -ivfsoverlay %t/vfs.yaml -E -C %s 2>&1 | FileCheck --check-prefix=DIRECT %s
// RUN: %clang_cc1 -Werror -I %t/Underlying -ivfsoverlay %t/vfs.yaml -DNESTED -E -C %s 2>&1 | FileCheck --check-prefix=DIRECT %s
// RUN: sed -e "s@INPUT_DIR@Overlay@g" -e "s@OUT_DIR@%{/t:regex_replacement}/Underlying@g" %S/Inputs/vfsoverlay-directory-relative.yaml > %t/vfs-relative.yaml
// RUN: %clang_cc1 -Werror -I %t/Underlying -ivfsoverlay %t/vfs-relative.yaml -E -C %s 2>&1 | FileCheck --check-prefix=DIRECT %s

// DIRECT: {{^}}// B.h in Underlying
// DIRECT: {{^}}// C.h in Overlay

// 2) Underlying -> Middle -> Overlay (C.h found, B.h falling back to Underlying)
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}/Overlay@g" -e "s@OUT_DIR@%{/t:regex_replacement}/Middle@g" %S/Inputs/vfsoverlay-directory.yaml > %t/vfs.yaml
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}/Middle@g" -e "s@OUT_DIR@%{/t:regex_replacement}/Underlying@g" %S/Inputs/vfsoverlay-directory.yaml > %t/vfs2.yaml
// RUN: %clang_cc1 -Werror -I %t/Underlying -ivfsoverlay %t/vfs.yaml -ivfsoverlay %t/vfs2.yaml -E -C %s 2>&1 | FileCheck --check-prefix=DIRECT %s
// RUN: %clang_cc1 -Werror -I %t/Underlying -ivfsoverlay %t/vfs.yaml -ivfsoverlay %t/vfs2.yaml -DNESTED -E -C %s 2>&1 | FileCheck --check-prefix=DIRECT %s

// Same as direct above

// 3) Underlying -> Middle -> Overlay (C.h falling back to Middle, B.h falling back to Underlying)
// RUN: rm -f %t/Overlay/C.h
// RUN: %clang_cc1 -Werror -I %t/Underlying -ivfsoverlay %t/vfs.yaml -ivfsoverlay %t/vfs2.yaml -E -C %s 2>&1 | FileCheck --check-prefix=FALLBACK %s

// FALLBACK: {{^}}// B.h in Underlying
// FALLBACK: {{^}}// C.h in Middle

// 3) Underlying -> Middle -> Overlay (C.h falling back to Underlying, B.h falling back to Underlying)
// RUN: rm -f %t/Middle/C.h
// RUN: %clang_cc1 -Werror -I %t/Underlying -ivfsoverlay %t/vfs.yaml -ivfsoverlay %t/vfs2.yaml -E -C %s 2>&1 | FileCheck --check-prefix=FALLBACK2 %s

// FALLBACK2: {{^}}// B.h in Underlying
// FALLBACK2: {{^}}// C.h in Underlying

#include "B.h"
#ifndef NESTED
#include "C.h"
#endif
