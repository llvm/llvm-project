# REQUIRES: x86, aarch64

# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o %t/main-arm64-macos.o %t/main.s
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-iossimulator -o %t/main-arm64-sim.o %t/main.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/main-x86_64-macos.o %t/main.s
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o %t/foo-arm64-macos.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-iossimulator -o %t/foo-arm64-sim.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/foo-x86_64-macos.o %t/foo.s

# Exhaustive test for:
# (x86_64-macos, arm64-macos, arm64-ios-simulator) x (default, -adhoc_codesign, -no_adhoc-codesign) x (execute, dylib, bundle)

# RUN: %lld -lSystem -arch x86_64 -execute -o %t/out %t/main-x86_64-macos.o
# RUN: llvm-objdump --macho --all-headers %t/out | FileCheck --check-prefix=NO-ADHOC %s
# RUN: %lld          -arch x86_64 -dylib   -o %t/out %t/foo-x86_64-macos.o
# RUN: llvm-objdump --macho --all-headers  %t/out| FileCheck --check-prefix=NO-ADHOC %s
# RUN: %lld          -arch x86_64 -bundle  -o %t/out %t/foo-x86_64-macos.o
# RUN: llvm-objdump --macho --all-headers  %t/out| FileCheck --check-prefix=NO-ADHOC %s

# RUN: %lld -lSystem -arch x86_64 -execute -adhoc_codesign -o %t/out %t/main-x86_64-macos.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s
# RUN: %lld          -arch x86_64 -dylib   -adhoc_codesign -o %t/out %t/foo-x86_64-macos.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s
# RUN: %lld          -arch x86_64 -bundle  -adhoc_codesign -o %t/out %t/foo-x86_64-macos.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s

# RUN: %lld -lSystem -arch x86_64 -execute -no_adhoc_codesign -o %t/out %t/main-x86_64-macos.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s
# RUN: %lld          -arch x86_64 -dylib   -no_adhoc_codesign -o %t/out %t/foo-x86_64-macos.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s
# RUN: %lld          -arch x86_64 -bundle  -no_adhoc_codesign -o %t/out %t/foo-x86_64-macos.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s


# RUN: %lld -lSystem -arch arm64 -execute -o %t/out %t/main-arm64-macos.o
# RUN: llvm-objdump --macho --all-headers %t/out | FileCheck --check-prefix=ADHOC %s
# RUN: %lld          -arch arm64 -dylib   -o %t/out %t/foo-arm64-macos.o
# RUN: llvm-objdump --macho --all-headers  %t/out| FileCheck --check-prefix=ADHOC %s
# RUN: %lld          -arch arm64 -bundle  -o %t/out %t/foo-arm64-macos.o
# RUN: llvm-objdump --macho --all-headers  %t/out| FileCheck --check-prefix=ADHOC %s

# RUN: %lld -lSystem -arch arm64 -execute -adhoc_codesign -o %t/out %t/main-arm64-macos.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s
# RUN: %lld          -arch arm64 -dylib   -adhoc_codesign -o %t/out %t/foo-arm64-macos.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s
# RUN: %lld          -arch arm64 -bundle  -adhoc_codesign -o %t/out %t/foo-arm64-macos.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s

# RUN: %lld -lSystem -arch arm64 -execute -no_adhoc_codesign -o %t/out %t/main-arm64-macos.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s
# RUN: %lld          -arch arm64 -dylib   -no_adhoc_codesign -o %t/out %t/foo-arm64-macos.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s
# RUN: %lld          -arch arm64 -bundle  -no_adhoc_codesign -o %t/out %t/foo-arm64-macos.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s


# RUN: %no-arg-lld -arch arm64 -platform_version ios-simulator 14.0 15.0 -execute -o %t/out %t/main-arm64-sim.o -syslibroot %S/Inputs/iPhoneSimulator.sdk -lSystem
# RUN: llvm-objdump --macho --all-headers %t/out | FileCheck --check-prefix=ADHOC %s
# RUN: %no-arg-lld -arch arm64 -platform_version ios-simulator 14.0 15.0 -dylib   -o %t/out %t/foo-arm64-sim.o
# RUN: llvm-objdump --macho --all-headers  %t/out| FileCheck --check-prefix=ADHOC %s
# RUN: %no-arg-lld -arch arm64 -platform_version ios-simulator 14.0 15.0 -bundle  -o %t/out %t/foo-arm64-sim.o
# RUN: llvm-objdump --macho --all-headers  %t/out| FileCheck --check-prefix=ADHOC %s

# RUN: %no-arg-lld -arch arm64 -platform_version ios-simulator 14.0 15.0 -execute -adhoc_codesign -o %t/out %t/main-arm64-sim.o -syslibroot %S/Inputs/iPhoneSimulator.sdk -lSystem
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s
# RUN: %no-arg-lld -arch arm64 -platform_version ios-simulator 14.0 15.0 -dylib   -adhoc_codesign -o %t/out %t/foo-arm64-sim.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s
# RUN: %no-arg-lld -arch arm64 -platform_version ios-simulator 14.0 15.0 -bundle  -adhoc_codesign -o %t/out %t/foo-arm64-sim.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s

# RUN: %no-arg-lld -lSystem -arch arm64 -platform_version ios-simulator 14.0 15.0 -execute -no_adhoc_codesign -o %t/out %t/main-arm64-sim.o -syslibroot %S/Inputs/iPhoneSimulator.sdk
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s
# RUN: %no-arg-lld -arch arm64 -platform_version ios-simulator 14.0 15.0 -dylib   -no_adhoc_codesign -o %t/out %t/foo-arm64-sim.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s
# RUN: %no-arg-lld -arch arm64 -platform_version ios-simulator 14.0 15.0 -bundle  -no_adhoc_codesign -o %t/out %t/foo-arm64-sim.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s

# RUN: %lld -arch x86_64 -dylib -o %t/out_installname.dylib -install_name @rpath/MyInstallName %t/foo-x86_64-macos.o -adhoc_codesign
# RUN: %lld -arch x86_64 -dylib -o %t/out_no_installname.dylib %t/foo-x86_64-macos.o -adhoc_codesign

## Smoke check to verify the dataoff and datasize value before using them with code-signature-check.py
# RUN: llvm-objdump --macho --all-headers %t/out_installname.dylib | FileCheck %s --check-prefix CS-ID-PRE -D#DATA_OFFSET=4176 -D#DATA_SIZE=192 
# RUN: llvm-objdump --macho --all-headers %t/out_no_installname.dylib | FileCheck %s --check-prefix CS-ID-PRE -D#DATA_OFFSET=4176 -D#DATA_SIZE=208 

## Verify that the 'Identifier' (aka 'Code Directory ID') field are set to the install-name, if available.
# RUN: %python %p/Inputs/code-signature-check.py %t/out_installname.dylib 4176 192 0 4176 | FileCheck %s --check-prefix CS-ID-INSTALL
# RUN:  %python %p/Inputs/code-signature-check.py %t/out_no_installname.dylib 4176 208 0 4176 | FileCheck %s --check-prefix CS-ID-NO-INSTALL

# ADHOC:          cmd LC_CODE_SIGNATURE
# ADHOC-NEXT: cmdsize 16

# NO-ADHOC-NOT:          cmd LC_CODE_SIGNATURE

# CS-ID-PRE: cmd LC_CODE_SIGNATURE
# CS-ID-PRE-NEXT: cmdsize 16
# CS-ID-PRE-NEXT: dataoff [[#DATA_OFFSET]]
# CS-ID-PRE-NEXT: datasize [[#DATA_SIZE]]

# CS-ID-INSTALL: Code Directory ID: MyInstallName
# CS-ID-NO-INSTALL: Code Directory ID: out_no_installname.dylib

#--- foo.s
.globl _foo
_foo:
  ret

#--- main.s
.globl _main
_main:
  ret
