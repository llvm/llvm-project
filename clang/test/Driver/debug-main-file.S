// RUN: %clang -### -c -save-temps -integrated-as -g %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: main-file-name

#ifdef(1)
foo:
        nop
        nop
        nop
#endif
