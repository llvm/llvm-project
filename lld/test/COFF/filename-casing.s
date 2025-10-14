# REQUIRES: x86

# RUN: mkdir -p %t.dir
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc -o %t.dir/MixedCase.obj %s
# RUN: not lld-link /entry:main %t.dir/MixedCase.obj 2>&1 | FileCheck -check-prefix=OBJECT %s

# RUN: llvm-lib /out:%t.dir/MixedCase.lib %t.dir/MixedCase.obj
# RUN: not lld-link /machine:x64 /entry:main %t.dir/MixedCase.lib 2>&1 | FileCheck -check-prefix=ARCHIVE %s

# OBJECT: undefined symbol: f
# OBJECT-NEXT: >>> referenced by {{.*}}MixedCase.obj:(main)
# ARCHIVE: undefined symbol: f
# ARCHIVE-NEXT: >>> referenced by {{.*}}MixedCase.lib(MixedCase.obj):(main)

.globl main
main:
	callq	f
