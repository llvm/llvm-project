# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t

## Ensure that weak externals are considered during subsystem inference.

# RUN: llvm-mc -triple x86_64-windows-msvc -filetype obj -o %t/cui.obj %t/cui.s
# RUN: lld-link -out:%t/cui.exe %t/cui.obj
# RUN: llvm-readobj --file-headers %t/cui.exe | FileCheck --check-prefix=CUI %s
# RUN: llvm-mc -triple x86_64-windows-msvc -filetype obj -o %t/gui.obj %t/gui.s
# RUN: lld-link -out:%t/gui.exe %t/gui.obj
# RUN: llvm-readobj --file-headers %t/gui.exe | FileCheck --check-prefix=GUI %s

# CUI:     Subsystem: IMAGE_SUBSYSTEM_WINDOWS_CUI
# GUI:     Subsystem: IMAGE_SUBSYSTEM_WINDOWS_GUI

#--- cui.s
.global default_main
default_main:
	ret

.weak main
main = default_main

.global mainCRTStartup
mainCRTStartup:
	ret

#--- gui.s
.global default_WinMain
default_WinMain:
	ret

.weak WinMain
WinMain = default_WinMain

.global WinMainCRTStartup
WinMainCRTStartup:
	ret
