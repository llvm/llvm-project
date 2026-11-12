# REQUIRES: x86

# RUN: llvm-mc -triple=x86_64-windows-gnu %s -filetype=obj -o %t.obj
# RUN: llvm-readobj --symbols %t.obj | FileCheck %s --check-prefix=SYMBOL

# RUN: lld-link -lldmingw -entry:main %t.obj -out:%t.exe -lldmap:%t.map -verbose
# RUN: llvm-readobj --sections %t.exe | FileCheck %s

# CHECK: Sections [
# CHECK:   Section {
# CHECK:     Number: 2
# CHECK-LABEL:     Name: .rdata (2E 72 64 61 74 61 00 00)
#             This is the critical check to show that .xdata$foo was
#             retained, while .xdata$bar wasn't. This *must* be 0x24
#             (0x4 for the .xdata section and 0x20 for the
#             .ctors/.dtors headers/ends).
# CHECK-NEXT:     VirtualSize: 0x24

# Check that the weak symbols still are emitted as it was when the test was
# written, to make sure the test still actually tests what was intended.

# SYMBOL:       Symbol {
# SYMBOL:         Name: foo
# SYMBOL-NEXT:    Value: 0
# SYMBOL-NEXT:    Section: IMAGE_SYM_UNDEFINED (0)
# SYMBOL-NEXT:    BaseType: Null (0x0)
# SYMBOL-NEXT:    ComplexType: Null (0x0)
# SYMBOL-NEXT:    StorageClass: WeakExternal (0x69)
# SYMBOL-NEXT:    AuxSymbolCount: 1
# SYMBOL-NEXT:    AuxWeakExternal {
# SYMBOL-NEXT:      Linked: .weak.foo.default.main (19)
# SYMBOL-NEXT:      Search: Alias (0x3)
# SYMBOL-NEXT:    }
# SYMBOL-NEXT:  }

        .text
        .globl          main
main:
        call            foo
        retq

# See associative-comdat-mingw.s for the general setup. Here, the leader
# symbols are weak, which causes the functions foo and bar to be undefined
# weak externals, while the actual leader symbols are named like
# .weak.foo.default.main.

        .section        .xdata$foo,"dr"
        .linkonce       discard
        .long           42

        .section        .xdata$bar,"dr"
        .linkonce       discard
        .long           43

        .section        .text$foo,"xr",discard,foo
        .weak           foo
foo:
        ret

        .section        .text$bar,"xr",discard,bar
        .weak           bar
bar:
        ret
