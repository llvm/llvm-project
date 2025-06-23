// RUN: %clang -O0 -g -debug-info-kind=standalone -dwarf-version=5 -fsanitize=nonnull-attribute \
// RUN: -fsanitize-trap=nonnull-attribute -emit-llvm -S -c %s -o - | FileCheck %s

__attribute__((nonnull))
void nonnull_arg(int *p) { 
    (void)p; 
}

void trigger_nonnull_arg()
{
    nonnull_arg(0);
}


// CHECK: call void @llvm.ubsantrap(i8 16) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: distinct !DISubprogram(name: "__clang_trap_msg$UBSan Trap Reason