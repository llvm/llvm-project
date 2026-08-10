// Tests that the z/OS toolchain adds system includes to its search path.

// RUN: %clang -c -### %s --target=s390x-ibm-zos 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "-D_UNIX03_WITHDRAWN"
// CHECK-SAME: "-D_OPEN_DEFAULT"
// CHECK-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include{{(/|\\\\)}}zos_wrappers"
// CHECK-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK-SAME: "-internal-isystem" "/usr/include"
// CHECK-SAME: "-fshort-enums"
// CHECK-SAME: "-fno-signed-char"
// CHECK-SAME: "-fno-signed-wchar"

// RUN: %clang -c -### -mzos-sys-include=/ABC/DEF %s 2>&1 \
// RUN:		--target=s390x-ibm-zos \
// RUN:   | FileCheck --check-prefixes=CHECK2 %s

// CHECK2: "-D_UNIX03_WITHDRAWN"
// CHECK2-SAME: "-D_OPEN_DEFAULT"
// CHECK2-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK2-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include{{(/|\\\\)}}zos_wrappers"
// CHECK2-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK2-SAME: "-internal-isystem" "/ABC/DEF"
// CHECK2-NOT: "-internal-isystem" "/usr/include"
// CHECK2-SAME: "-fshort-enums"
// CHECK2-SAME: "-fno-signed-char"
// CHECK2-SAME: "-fno-signed-wchar"

// RUN: %clang -c -### -mzos-sys-include=/ABC/DEF:/ghi/jkl %s 2>&1 \
// RUN:		--target=s390x-ibm-zos \
// RUN:   | FileCheck --check-prefixes=CHECK3 %s

// CHECK3: "-D_UNIX03_WITHDRAWN"
// CHECK3-SAME: "-D_OPEN_DEFAULT"
// CHECK3-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK3-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include{{(/|\\\\)}}zos_wrappers"
// CHECK3-SAME: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK3-SAME: "-internal-isystem" "/ABC/DEF"
// CHECK3-SAME: "-internal-isystem" "/ghi/jkl"
// CHECK3-NOT: "-internal-isystem" "/usr/include"
// CHECK3-SAME: "-fshort-enums"
// CHECK3-SAME: "-fno-signed-char"
// CHECK3-SAME: "-fno-signed-wchar"

// RUN: %clang -c -### -nostdinc %s 2>&1 \
// RUN:		--target=s390x-ibm-zos \
// RUN:   | FileCheck --check-prefixes=CHECK4 %s

// CHECK4: "-D_UNIX03_WITHDRAWN"
// CHECK4-SAME: "-D_OPEN_DEFAULT"
// CHECK4-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK4-NOT: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include{{(/|\\\\)}}zos_wrappers"
// CHECK4-NOT: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK4-NOT: "-internal-isystem" "/usr/include"
// CHECK4-SAME: "-fshort-enums"
// CHECK4-SAME: "-fno-signed-char"
// CHECK4-SAME: "-fno-signed-wchar"


// RUN: %clang -c -### -nobuiltininc %s 2>&1 \
// RUN:		--target=s390x-ibm-zos \
// RUN:   | FileCheck --check-prefixes=CHECK5 %s

// CHECK5: "-D_UNIX03_WITHDRAWN"
// CHECK5-SAME: "-D_OPEN_DEFAULT"
// CHECK5-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK5-NOT: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include{{(/|\\\\)}}zos_wrappers"
// CHECK5-NOT: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// CHECK5-SAME: "-internal-isystem" "/usr/include"
// CHECK5-SAME: "-fshort-enums"
// CHECK5-SAME: "-fno-signed-char"
// CHECK5-SAME: "-fno-signed-wchar"

