// RUN: %check_clang_tidy -std=c++20-or-later %s misc-use-internal-linkage %t

// Symbols in a partition are visible to any TU in the same module 
// that imports that partition, so we shouldn't warn on them.

module foo:bar;

void fn_in_partition() {}
int var_in_partition;
struct StructInPartition {};
