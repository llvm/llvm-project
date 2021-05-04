// Verify ubsan doesn't emit checks for blacklisted functions and files
// RUN: echo "fun:hash" > %t-func.blacklist
// RUN: echo "src:%s" | sed -e 's/\\/\\\\/g' > %t-file.blacklist

// RUN: rm -f %t-vfsoverlay.yaml
// RUN: rm -f %t-nonexistent.blacklist
// RUN: sed -e "s|@DIR@|%/T|g" %S/Inputs/sanitizer-blacklist-vfsoverlay.yaml | sed -e "s|@REAL_FILE@|%/t-func.blacklist|g" | sed -e "s|@NONEXISTENT_FILE@|%/t-nonexistent.blacklist|g" > %t-vfsoverlay.yaml
// RUN: %clang_cc1 -fsanitize=unsigned-integer-overflow -ivfsoverlay %t-vfsoverlay.yaml -fsanitize-blacklist=%/T/only-virtual-file.blacklist -emit-llvm %s -o - | FileCheck %s --check-prefix=FUNC

// RUN: not %clang_cc1 -fsanitize=unsigned-integer-overflow -ivfsoverlay %t-vfsoverlay.yaml -fsanitize-blacklist=%/T/invalid-virtual-file.blacklist -emit-llvm %s -o - 2>&1 | FileCheck -DMSG=%errc_ENOENT %s --check-prefix=INVALID-MAPPED-FILE
// INVALID-MAPPED-FILE: invalid-virtual-file.blacklist': [[MSG]]

// RUN: not %clang_cc1 -fsanitize=unsigned-integer-overflow -ivfsoverlay %t-vfsoverlay.yaml -fsanitize-blacklist=%t-nonexistent.blacklist -emit-llvm %s -o - 2>&1 | FileCheck -DMSG=%errc_ENOENT %s --check-prefix=INVALID
// INVALID: nonexistent.blacklist': [[MSG]]

unsigned i;

// DEFAULT: @hash
// FUNC: @hash
// FILE: @hash
unsigned hash() {
// DEFAULT: call {{.*}}void @__ubsan
// FUNC-NOT: call {{.*}}void @__ubsan
// FILE-NOT: call {{.*}}void @__ubsan
  return i * 37;
}

// DEFAULT: @add
// FUNC: @add
// FILE: @add
unsigned add() {
// DEFAULT: call {{.*}}void @__ubsan
// FUNC: call {{.*}}void @__ubsan
// FILE-NOT: call {{.*}}void @__ubsan
  return i + 1;
}
