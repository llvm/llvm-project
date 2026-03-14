// Tests that the z/OS toolchain adds system includes to its search path.

// RUN: %clangxx -c -### %s --target=s390x-ibm-zos 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "-D_UNIX03_WITHDRAWN"
// CHECK-SAME: "-D_OPEN_DEFAULT"
// CHECK-SAME: "-D_XOPEN_SOURCE=600"
// CHECK-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-SAME: "-internal-isystem" "{{.*}}{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include{{(/|\\\\)}}zos_wrappers"
// CHECK-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK-SAME: "-internal-isystem" "/usr/include"
// CHECK-SAME: "-fshort-enums"
// CHECK-SAME: "-fno-signed-char"
// CHECK-SAME: "-fno-signed-wchar"

// RUN: %clangxx -c -### -mzos-sys-include=/ABC/DEF %s 2>&1 \
// RUN:		--target=s390x-ibm-zos \
// RUN:   | FileCheck --check-prefixes=CHECK2 %s

// CHECK2: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK2-SAME: "-internal-isystem" "{{.*}}{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK2-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include{{(/|\\\\)}}zos_wrappers"
// CHECK2-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK2-SAME: "-internal-isystem" "{{/|\\\\}}ABC{{/|\\\\}}DEF"
// CHECK2-NOT: "-internal-isystem" "/usr/include"
// CHECK2-SAME: "-fshort-enums"
// CHECK2-SAME: "-fno-signed-char"
// CHECK2-SAME: "-fno-signed-wchar"

// RUN: %clangxx -c -### -nostdinc %s 2>&1 \
// RUN:		--target=s390x-ibm-zos \
// RUN:   | FileCheck --check-prefixes=CHECK3 %s

// CHECK3: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK3-NOT: "-internal-isystem" "{{.*}}/bin/../include/c++/v1"
// CHECK3-NOT: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include{{(/|\\\\)}}zos_wrappers"
// CHECK3-NOT: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK3-NOT: "-internal-isystem" "/usr/include"
// CHECK3-SAME: "-fshort-enums"
// CHECK3-SAME: "-fno-signed-char"
// CHECK3-SAME: "-fno-signed-wchar"

// RUN: %clangxx -c -### -nostdinc++ %s 2>&1 \
// RUN:		--target=s390x-ibm-zos \
// RUN:   | FileCheck --check-prefixes=CHECK4 %s

// CHECK4: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK4-NOT: "-internal-isystem" "{{.*}}{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK4-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include{{(/|\\\\)}}zos_wrappers"
// CHECK4-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK4-SAME: "-internal-isystem" "/usr/include"
// CHECK4-SAME: "-fshort-enums"
// CHECK4-SAME: "-fno-signed-char"
// CHECK4-SAME: "-fno-signed-wchar"

// RUN: %clangxx -c -### -nostdlibinc %s 2>&1 \
// RUN:		--target=s390x-ibm-zos \
// RUN:   | FileCheck --check-prefixes=CHECK5 %s

// CHECK5: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK5-NOT: "-internal-isystem" "{{.*}}{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK5-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include{{(/|\\\\)}}zos_wrappers"
// CHECK5-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK5-SAME: "-internal-isystem" "/usr/include"
// CHECK5-SAME: "-fshort-enums"
// CHECK5-SAME: "-fno-signed-char"
// CHECK5-SAME: "-fno-signed-wchar"

// RUN: %clangxx -c -### -nobuiltininc %s 2>&1 \
// RUN:		--target=s390x-ibm-zos \
// RUN:   | FileCheck --check-prefixes=CHECK6 %s

// CHECK6: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK6-SAME: "-internal-isystem" "{{.*}}{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK6-NOT: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include{{(/|\\\\)}}zos_wrappers"
// CHECK6-NOT: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK6-SAME: "-internal-isystem" "/usr/include"
// CHECK6-SAME: "-fshort-enums"
// CHECK6-SAME: "-fno-signed-char"
// CHECK6-SAME: "-fno-signed-wchar"

// RUN: %clangxx -c -### -D_XOPEN_SOURCE=700 %s 2>&1 \
// RUN:		--target=s390x-ibm-zos \
// RUN:   | FileCheck --check-prefixes=CHECK7 %s

// CHECK7: "-D_UNIX03_WITHDRAWN"
// CHECK7-SAME: "-D_OPEN_DEFAULT"
// CHECK7-NOT: "-D_XOPEN_SOURCE=600"
// CHECK7-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK7-SAME: "-D" "_XOPEN_SOURCE=700"
// CHECK7-SAME: "-internal-isystem" "{{.*}}{{/|\\\\}}bin{{/|\\\\}}..{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK7-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include{{(/|\\\\)}}zos_wrappers"
// CHECK7-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK7-SAME: "-internal-isystem" "/usr/include"
// CHECK7-SAME: "-fshort-enums"
// CHECK7-SAME: "-fno-signed-char"
// CHECK7-SAME: "-fno-signed-wchar"
