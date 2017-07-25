// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-linux-gnu -O1 -S -o - %s | FileCheck -check-prefix=STRCPY -check-prefix=MEMSET %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fno-builtin -O1 -S -o - %s | FileCheck -check-prefix=NOSTRCPY -check-prefix=NOMEMSET %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fno-builtin-memset -O1 -S -o - %s | FileCheck -check-prefix=STRCPY -check-prefix=NOMEMSET %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -O1 -fexperimental-new-pass-manager -S -o - %s | FileCheck -check-prefix=STRCPY -check-prefix=MEMSET %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fno-builtin -O1 -fexperimental-new-pass-manager -S -o - %s | FileCheck -check-prefix=NOSTRCPY -check-prefix=NOMEMSET %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fno-builtin-memset -O1 -fexperimental-new-pass-manager -S -o - %s | FileCheck -check-prefix=STRCPY -check-prefix=NOMEMSET %s

void PR13497() {
  char content[2];
  // make sure we don't optimize this call to strcpy()
  // STRCPY-NOT: __strcpy_chk
  // NOSTRCPY: __strcpy_chk
  __builtin___strcpy_chk(content, "", 1);
}

void PR4941(char *s) {
  // Make sure we don't optimize this loop to a memset().
  // NOMEMSET-NOT: memset
  // MEMSET: memset
  for (unsigned i = 0; i < 8192; ++i)
    s[i] = 0;
}
