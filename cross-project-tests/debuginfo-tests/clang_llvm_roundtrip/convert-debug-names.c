// REQUIRES: x86-registered-target

// Base types synthesized for DW_OP_convert are named DW_TAG_base_type DIEs, so
// they must be indexed in .debug_names like any other named type.

// RUN: %clang --target=x86_64-linux -c -g -O2 -gdwarf-5 -gpubnames %s -o %t.o
// RUN: llvm-dwarfdump --verify %t.o | FileCheck %s --check-prefix=VERIFY
// RUN: llvm-dwarfdump --find=DW_ATE_signed_8 --find=DW_ATE_signed_32 \
// RUN:   --find=DW_ATE_unsigned_16 --find=DW_ATE_unsigned_32 %t.o | \
// RUN:   FileCheck %s --check-prefix=FIND

// VERIFY: No errors.

// FIND-DAG: DW_AT_name ("DW_ATE_signed_8")
// FIND-DAG: DW_AT_name ("DW_ATE_signed_32")
// FIND-DAG: DW_AT_name ("DW_ATE_unsigned_16")
// FIND-DAG: DW_AT_name ("DW_ATE_unsigned_32")

int use(int);

// The optimizer drops the unused extensions and describes the variables with
// DW_OP_convert on the incoming arguments.
int f(signed char c, unsigned short u) {
  int widened = c;
  unsigned int zwidened = u;
  return use(0);
}
