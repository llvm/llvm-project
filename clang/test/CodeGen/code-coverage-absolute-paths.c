// RUN: rm -rf %t && mkdir %t
// RUN: cd %S

// RUN: %clang_cc1 -emit-llvm-bc -o /dev/null -coverage-notes-file=%t/notes-relative.gcno %{s:basename}
// RUN: FileCheck --check-prefix=RELATIVE %s --input-file %t/notes-relative.gcno -DABSOLUTE_PATH=%s

// RUN: %clang_cc1 -emit-llvm-bc -o /dev/null -coverage-notes-file=%t/notes-absolute.gcno -coverage-notes-abs-paths %{s:basename}
// RUN: FileCheck --check-prefix=ABSOLUTE %s --input-file %t/notes-absolute.gcno -DABSOLUTE_PATH=%s

void test() {}

// RELATIVE-NOT: [[ABSOLUTE_PATH]]
// ABSOLUTE: [[ABSOLUTE_PATH]]
