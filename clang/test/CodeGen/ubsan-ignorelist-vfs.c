// Verify ubsan doesn't emit checks for ignorelisted functions and files
// RUN: echo "fun:hash" > %t-func.ignorelist
// RUN: echo "src:%s" | sed -e 's/\\/\\\\/g' > %t-file.ignorelist

// RUN: rm -f %t-vfsoverlay.yaml
// RUN: rm -f %t-nonexistent.ignorelist
// RUN: sed -e "s|@DIR@|%/T|g" %S/Inputs/sanitizer-ignorelist-vfsoverlay.yaml | sed -e "s|@REAL_FILE@|%/t-func.ignorelist|g" | sed -e "s|@NONEXISTENT_FILE@|%/t-nonexistent.ignorelist|g" > %t-vfsoverlay.yaml
// RUN: %clang_cc1 -fsanitize=unsigned-integer-overflow -ivfsoverlay %t-vfsoverlay.yaml -fsanitize-ignorelist=%/T/only-virtual-file.ignorelist -emit-llvm %s -o - | FileCheck %s --check-prefix=FUNC

// RUN: not %clang_cc1 -fsanitize=unsigned-integer-overflow -ivfsoverlay %t-vfsoverlay.yaml -fsanitize-ignorelist=%/T/invalid-virtual-file.ignorelist -emit-llvm %s -o - 2>&1 | FileCheck -DMSG=%errc_ENOENT %s --check-prefix=INVALID-MAPPED-FILE
// INVALID-MAPPED-FILE: invalid-virtual-file.ignorelist': [[MSG]]

// RUN: not %clang_cc1 -fsanitize=unsigned-integer-overflow -ivfsoverlay %t-vfsoverlay.yaml -fsanitize-ignorelist=%t-nonexistent.ignorelist -emit-llvm %s -o - 2>&1 | FileCheck -DMSG=%errc_ENOENT %s --check-prefix=INVALID
// INVALID: nonexistent.ignorelist': [[MSG]]

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
