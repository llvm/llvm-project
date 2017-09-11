# REQEUIRES: x86
# RUN: rm -f a.exe a.dll

# RUN: llvm-mc -triple=x86_64-windows-msvc %s -filetype=obj -o %t.obj
# RUN: ld.lld -m i386pep --entry main %t.obj
# RUN: llvm-readobj a.exe | FileCheck %s
# RUN: ld.lld -m i386pep -shared --entry main %t.obj
# RUN: llvm-readobj a.dll | FileCheck %s
# RUN: ld.lld -m i386pep -e main %t.obj -o %t.exe
# RUN: llvm-readobj %t.exe | FileCheck %s
# CHECK: File:

# RUN: ld.lld -m i386pep -e main %t.obj -o %t.exe -verbose -### | FileCheck %s -check-prefix CHECK-VERBOSE
# CHECK-VERBOSE: lld-link -entry:main
# CHECK-VERBOSE-SAME: -machine:x64 -alternatename:__image_base__=__ImageBase
# CHECK-VERBOSE-SAME: -verbose

# RUN: ld.lld -m i386pep --entry main %t.obj -o %t.exe --subsystem console
# RUN: llvm-readobj -file-headers %t.exe | FileCheck %s -check-prefix CHECK-CONSOLE
# CHECK-CONSOLE: Subsystem: IMAGE_SUBSYSTEM_WINDOWS_CUI (0x3)

# RUN: ld.lld -m i386pep --entry main %t.obj -o %t.exe --subsystem windows
# RUN: llvm-readobj -file-headers %t.exe | FileCheck %s -check-prefix CHECK-WINDOWS
# CHECK-WINDOWS: Subsystem: IMAGE_SUBSYSTEM_WINDOWS_GUI (0x2)

# RUN: ld.lld -m i386pep --entry main %t.obj -o %t.exe --stack 4194304,8192
# RUN: llvm-readobj -file-headers %t.exe | FileCheck %s -check-prefix CHECK-STACK
# CHECK-STACK: SizeOfStackReserve: 4194304
# CHECK-STACK: SizeOfStackCommit: 8192

# RUN: yaml2obj < %p/Inputs/imagebase-i386.yaml > %t.obj
# RUN: ld.lld -m i386pe %t.obj -o %t.exe
# RUN: llvm-readobj -file-headers %t.exe | FileCheck %s -check-prefix CHECK-I386
# CHECK-I386: Machine: IMAGE_FILE_MACHINE_I386

# RUN: yaml2obj < %p/Inputs/imagebase-x86_64.yaml > %t.obj
# RUN: ld.lld -m i386pep %t.obj -o %t.exe
# RUN: llvm-readobj -file-headers %t.exe | FileCheck %s -check-prefix CHECK-AMD64
# CHECK-AMD64: Machine: IMAGE_FILE_MACHINE_AMD64

# RUN: yaml2obj < %p/Inputs/imagebase-arm.yaml > %t.obj
# RUN: ld.lld -m thumb2pe %t.obj -o %t.exe
# RUN: llvm-readobj -file-headers %t.exe | FileCheck %s -check-prefix CHECK-ARMNT
# CHECK-ARMNT: Machine: IMAGE_FILE_MACHINE_ARMNT

# RUN: yaml2obj < %p/Inputs/imagebase-aarch64.yaml > %t.obj
# RUN: ld.lld -m arm64pe %t.obj -o %t.exe
# RUN: llvm-readobj -file-headers %t.exe | FileCheck %s -check-prefix CHECK-ARM64
# CHECK-ARM64: Machine: IMAGE_FILE_MACHINE_ARM64

# RUN: yaml2obj < %p/../COFF/Inputs/export.yaml > %t.obj
# RUN: ld.lld -m i386pep --shared %t.obj -o %t.dll --out-implib %t.lib
# RUN: llvm-readobj %t.lib | FileCheck %s -check-prefix CHECK-IMPLIB
# CHECK-IMPLIB: Symbol: __imp_exportfn3
# CHECK-IMPLIB: Symbol: exportfn3

.global main
.text
main:
  ret
