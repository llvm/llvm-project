// RUN: %clang_cc1 -fno-builtin -O1 -S -o - %s | FileCheck %s
// RUN: %clang_cc1 -fno-builtin-memset -O1 -S -o - %s | FileCheck -check-prefix=MEMSET %s

void PR13497() {
  char content[2];
  // make sure we don't optimize this call to strcpy()
  // CHECK: __strcpy_chk
  __builtin___strcpy_chk(content, "", 1);
}

void PR4941(char *s) {
  // Make sure we don't optimize this loop to a memset().
  // MEMSET-LABEL: PR4941:
  // MEMSET-NOT: memset
  for (unsigned i = 0; i < 8192; ++i)
    s[i] = 0;
}
