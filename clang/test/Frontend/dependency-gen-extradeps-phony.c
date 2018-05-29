// RUN: %clang -MM -MP -Xclang -fdepfile-entry=1.extra -Xclang -fdepfile-entry=2.extra -Xclang -fdepfile-entry=2.extra %s | \
// RUN: FileCheck %s --implicit-check-not=.c:
//
// Verify that phony targets are only created for the extra dependency files,
// and not the input file.

// CHECK: dependency-gen-extradeps-phony.o: 1.extra 2.extra \
// CHECK-NEXT: dependency-gen-extradeps-phony.c
// CHECK: 1.extra:
// CHECK: 2.extra:
