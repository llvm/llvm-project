// Check that ld gets arch_multiple.

// RUN: %clang -target i386-apple-darwin9 -arch i386 -arch x86_64 %s -### -o foo 2> %t.log
// RUN: grep '".*ld.*" .*"-arch_multiple" "-final_output" "foo"' %t.log

// Make sure we run dsymutil on source input files.
// RUN: %clang -target i386-apple-darwin9 -### -g %s -o BAR 2> %t.log
// RUN: grep -E '".*dsymutil(\.exe)?" "-o" "BAR.dSYM" "BAR"' %t.log
// RUN: %clang -target i386-apple-darwin9 -### -g -filelist FOO %s -o BAR 2> %t.log
// RUN: grep -E '".*dsymutil(\.exe)?" "-o" "BAR.dSYM" "BAR"' %t.log

// Check linker changes that came with new linkedit format.
// RUN: touch %t.o
// RUN: %clang -target i386-apple-darwin9 -fuse-ld= -mlinker-version=400 -### -arch armv6 -miphoneos-version-min=3.0 %t.o 2> %t.log
// RUN: %clang -target i386-apple-darwin9 -fuse-ld= -mlinker-version=400 -### -arch armv6 -miphoneos-version-min=3.0 -dynamiclib %t.o 2>> %t.log
// RUN: %clang -target i386-apple-darwin9 -fuse-ld= -mlinker-version=400 -### -arch armv6 -miphoneos-version-min=3.0 -bundle %t.o 2>> %t.log
// RUN: FileCheck -check-prefix=LINK_IPHONE_3_0 %s < %t.log

// LINK_IPHONE_3_0: {{ld(.exe)?"}}
// LINK_IPHONE_3_0: -iphoneos_version_min
// LINK_IPHONE_3_0: 3.0.0
// LINK_IPHONE_3_0-NOT: -lcrt1.3.1.o
// LINK_IPHONE_3_0: -lcrt1.o
// LINK_IPHONE_3_0: -lSystem
// LINK_IPHONE_3_0: {{ld(.exe)?"}}
// LINK_IPHONE_3_0: -dylib
// LINK_IPHONE_3_0: -ldylib1.o
// LINK_IPHONE_3_0: -lSystem
// LINK_IPHONE_3_0: {{ld(.exe)?"}}
// LINK_IPHONE_3_0: -lbundle1.o
// LINK_IPHONE_3_0: -lSystem

// RUN: %clang -target i386-apple-darwin9 -fuse-ld= -mlinker-version=400 -### -arch armv7 -miphoneos-version-min=3.1 %t.o 2> %t.log
// RUN: %clang -target i386-apple-darwin9 -fuse-ld= -mlinker-version=400 -### -arch armv7 -miphoneos-version-min=3.1 -dynamiclib %t.o 2>> %t.log
// RUN: %clang -target i386-apple-darwin9 -fuse-ld= -mlinker-version=400 -### -arch armv7 -miphoneos-version-min=3.1 -bundle %t.o 2>> %t.log
// RUN: FileCheck -check-prefix=LINK_IPHONE_3_1 %s < %t.log

// LINK_IPHONE_3_1: {{ld(.exe)?"}}
// LINK_IPHONE_3_1: -iphoneos_version_min
// LINK_IPHONE_3_1: 3.1.0
// LINK_IPHONE_3_1-NOT: -lcrt1.o
// LINK_IPHONE_3_1: -lcrt1.3.1.o
// LINK_IPHONE_3_1: -lSystem
// LINK_IPHONE_3_1: {{ld(.exe)?"}}
// LINK_IPHONE_3_1: -dylib
// LINK_IPHONE_3_1-NOT: -ldylib1.o
// LINK_IPHONE_3_1: -lSystem
// LINK_IPHONE_3_1: {{ld(.exe)?"}}
// LINK_IPHONE_3_1-NOT: -lbundle1.o
// LINK_IPHONE_3_1: -lSystem

// RUN: %clang -target i386-apple-darwin9 -fuse-ld= -mlinker-version=400 -### -arch i386 -mios-simulator-version-min=3.0 %t.o 2> %t.log
// RUN: %clang -target i386-apple-darwin9 -fuse-ld= -mlinker-version=400 -### -arch i386 -mios-simulator-version-min=3.0 -dynamiclib %t.o 2>> %t.log
// RUN: %clang -target i386-apple-darwin9 -fuse-ld= -mlinker-version=400 -### -arch i386 -mios-simulator-version-min=3.0 -bundle %t.o 2>> %t.log
// RUN: FileCheck -check-prefix=LINK_IOSSIM_3_0 %s < %t.log

// LINK_IOSSIM_3_0: {{ld(.exe)?"}}
// LINK_IOSSIM_3_0: -ios_simulator_version_min
// LINK_IOSSIM_3_0: 3.0.0
// LINK_IOSSIM_3_0-NOT: -lcrt1.o
// LINK_IOSSIM_3_0: -lSystem
// LINK_IOSSIM_3_0: {{ld(.exe)?"}}
// LINK_IOSSIM_3_0: -dylib
// LINK_IOSSIM_3_0-NOT: -ldylib1.o
// LINK_IOSSIM_3_0: -lSystem
// LINK_IOSSIM_3_0: {{ld(.exe)?"}}
// LINK_IOSSIM_3_0-NOT: -lbundle1.o
// LINK_IOSSIM_3_0: -lSystem

// RUN: %clang -target i386-apple-darwin9 -### -fpie %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_EXPLICIT_PIE %s < %t.log
//
// LINK_EXPLICIT_PIE: {{ld(.exe)?"}}
// LINK_EXPLICIT_PIE: "-pie"

// RUN: %clang -target i386-apple-darwin9 -### -fno-pie %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_EXPLICIT_NO_PIE %s < %t.log
//
// LINK_EXPLICIT_NO_PIE: {{ld(.exe)?"}}
// LINK_EXPLICIT_NO_PIE: "-no_pie"

// RUN: %clang -target x86_64-apple-darwin10 -### %t.o \
// RUN:   -fuse-ld= -mlinker-version=100 2> %t.log
// RUN: FileCheck -check-prefix=LINK_NEWER_DEMANGLE %s < %t.log
//
// LINK_NEWER_DEMANGLE: {{ld(.exe)?"}}
// LINK_NEWER_DEMANGLE: "-demangle"

// RUN: %clang -target x86_64-apple-darwin10 -### %t.o \
// RUN:   -fuse-ld= -mlinker-version=100 -Wl,--no-demangle 2> %t.log
// RUN: FileCheck -check-prefix=LINK_NEWER_NODEMANGLE %s < %t.log
//
// LINK_NEWER_NODEMANGLE: {{ld(.exe)?"}}
// LINK_NEWER_NODEMANGLE-NOT: "-demangle"
// LINK_NEWER_NODEMANGLE: "-lSystem"

// RUN: %clang -target x86_64-apple-darwin10 -### %t.o \
// RUN:   -fuse-ld= -mlinker-version=95 2> %t.log
// RUN: FileCheck -check-prefix=LINK_OLDER_NODEMANGLE %s < %t.log
//
// LINK_OLDER_NODEMANGLE: {{ld(.exe)?"}}
// LINK_OLDER_NODEMANGLE-NOT: "-demangle"
// LINK_OLDER_NODEMANGLE: "-lSystem"

// RUN: %clang -target x86_64-apple-darwin10 -### %s \
// RUN:   -fuse-ld= -mlinker-version=117 -flto 2> %t.log
// RUN: cat %t.log
// RUN: FileCheck -check-prefix=LINK_OBJECT_LTO_PATH %s < %t.log
//
// LINK_OBJECT_LTO_PATH: {{ld(.exe)?"}}
// LINK_OBJECT_LTO_PATH: "-object_path_lto"

// RUN: %clang -target x86_64-apple-darwin10 -### %t.o \
// RUN:   -force_load a -force_load b 2> %t.log
// RUN: cat %t.log
// RUN: FileCheck -check-prefix=FORCE_LOAD %s < %t.log
//
// FORCE_LOAD: {{ld(.exe)?"}}
// FORCE_LOAD: "-force_load" "a" "-force_load" "b"

// RUN: %clang -target x86_64-apple-darwin10 -### %t.o \
// RUN:   -lazy_framework Framework 2> %t.log
//
// RUN: FileCheck -check-prefix=LINK_LAZY_FRAMEWORK %s < %t.log
// LINK_LAZY_FRAMEWORK: {{ld(.exe)?"}}
// LINK_LAZY_FRAMEWORK: "-lazy_framework" "Framework"

// RUN: %clang -target x86_64-apple-darwin10 -### %t.o \
// RUN:   -lazy_library Library 2> %t.log
//
// RUN: FileCheck -check-prefix=LINK_LAZY_LIBRARY %s < %t.log
// LINK_LAZY_LIBRARY: {{ld(.exe)?"}}
// LINK_LAZY_LIBRARY: "-lazy_library" "Library"

// RUN: %clang -target x86_64-apple-darwin10 -fuse-ld= -mlinker-version=400 -### %t.o 2> %t.log
// RUN: %clang -target x86_64-apple-macosx10.7 -fuse-ld= -mlinker-version=400 -### %t.o 2>> %t.log
// RUN: FileCheck -check-prefix=LINK_VERSION_MIN %s < %t.log
// LINK_VERSION_MIN: {{ld(.exe)?"}}
// LINK_VERSION_MIN: "-macosx_version_min" "10.6.0"
// LINK_VERSION_MIN: {{ld(.exe)?"}}
// LINK_VERSION_MIN: "-macosx_version_min" "10.7.0"

// RUN: %clang -target x86_64-apple-ios13.1-macabi -fuse-ld= -mlinker-version=400 -### %t.o 2>> %t.log
// RUN: FileCheck -check-prefix=LINK_VERSION_MIN_MACABI %s < %t.log
// LINK_VERSION_MIN_MACABI: {{ld(.exe)?"}}
// LINK_VERSION_MIN_MACABI: "-maccatalyst_version_min" "13.1.0"
// LINK_VERSION_MIN_MACABI-NOT: macosx_version_min
// LINK_VERSION_MIN_MACABI-NOT: macos_version_min

// RUN: %clang -target x86_64-apple-darwin12 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_NO_CRT1 %s < %t.log
// LINK_NO_CRT1-NOT: crt

// RUN: %clang -target armv7-apple-ios6.0 -miphoneos-version-min=6.0 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_NO_IOS_CRT1 %s < %t.log
// LINK_NO_IOS_CRT1-NOT: crt

// RUN: %clang -target arm64-apple-ios5.0 -miphoneos-version-min=5.0 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_NO_IOS_ARM64_CRT1 %s < %t.log
// LINK_NO_IOS_ARM64_CRT1-NOT: crt

// RUN: %clang -target x86_64-apple-ios6.0 -miphoneos-version-min=6.0 -fprofile-instr-generate -resource-dir=%S/Inputs/resource_dir -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_IOSSIM_PROFILE %s < %t.log
// LINK_IOSSIM_PROFILE: {{ld(.exe)?"}}
// LINK_IOSSIM_PROFILE: libclang_rt.profile_iossim.a
// LINK_IOSSIM_PROFILE: libclang_rt.iossim.a

// RUN: %clang -target arm64-apple-tvos8.3 -fuse-ld= -mlinker-version=400 -mtvos-version-min=8.3 -resource-dir=%S/Inputs/resource_dir -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_TVOS_ARM64 %s < %t.log
// LINK_TVOS_ARM64: {{ld(.exe)?"}}
// LINK_TVOS_ARM64: -tvos_version_min
// LINK_TVOS_ARM64-NOT: crt
// LINK_TVOS_ARM64-NOT: lgcc_s.1
// LINK_TVOS_ARM64: libclang_rt.tvos.a

// RUN: %clang -target arm64-apple-tvos8.3 -fuse-ld= -mlinker-version=400 -mtvos-version-min=8.3 -fprofile-instr-generate -resource-dir=%S/Inputs/resource_dir  -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_TVOS_PROFILE %s < %t.log
// LINK_TVOS_PROFILE: {{ld(.exe)?"}}
// LINK_TVOS_PROFILE: libclang_rt.profile_tvos.a
// LINK_TVOS_PROFILE: libclang_rt.tvos.a

// RUN: %clang -target arm64-apple-tvos8.3 -fuse-ld= -mlinker-version=400 -mtvos-version-min=8.3 -resource-dir=%S/Inputs/resource_dir -### %t.o -lcc_kext 2> %t.log
// RUN: FileCheck -check-prefix=LINK_TVOS_KEXT %s < %t.log
// LINK_TVOS_KEXT: {{ld(.exe)?"}}
// LINK_TVOS_KEXT: libclang_rt.cc_kext_tvos.a
// LINK_TVOS_KEXT: libclang_rt.tvos.a

// RUN: %clang -target armv7k-apple-watchos2.0 -fuse-ld= -mlinker-version=400 -mwatchos-version-min=2.0 -resource-dir=%S/Inputs/resource_dir -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_WATCHOS_ARM %s < %t.log
// LINK_WATCHOS_ARM: {{ld(.exe)?"}}
// LINK_WATCHOS_ARM: -watchos_version_min
// LINK_WATCHOS_ARM-NOT: crt
// LINK_WATCHOS_ARM-NOT: lgcc_s.1
// LINK_WATCHOS_ARM: libclang_rt.watchos.a

// RUN: %clang -target armv7k-apple-watchos2.0 -fuse-ld= -mlinker-version=400 -mwatchos-version-min=2.0 -resource-dir=%S/Inputs/resource_dir -fprofile-instr-generate -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_WATCHOS_PROFILE %s < %t.log
// LINK_WATCHOS_PROFILE: {{ld(.exe)?"}}
// LINK_WATCHOS_PROFILE: libclang_rt.profile_watchos.a
// LINK_WATCHOS_PROFILE: libclang_rt.watchos.a

// RUN: %clang -target armv7k-apple-watchos2.0 -fuse-ld= -mlinker-version=400 -mwatchos-version-min=2.0 -resource-dir=%S/Inputs/resource_dir -### %t.o -lcc_kext 2> %t.log
// RUN: FileCheck -check-prefix=LINK_WATCHOS_KEXT %s < %t.log
// LINK_WATCHOS_KEXT: {{ld(.exe)?"}}
// LINK_WATCHOS_KEXT: libclang_rt.cc_kext_watchos.a
// LINK_WATCHOS_KEXT: libclang_rt.watchos.a

// RUN: %clang -target i386-apple-darwin12 -pg -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_PG %s < %t.log
// LINK_PG: -lgcrt1.o
// LINK_PG: -no_new_main

// RUN: %clang -target i386-apple-darwin13 -pg -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_PG_NO_SUPPORT_OSX %s < %t.log
// LINK_PG_NO_SUPPORT_OSX: error: the clang compiler does not support -pg option on versions of OS X

// RUN: %clang -target x86_64-apple-ios5.0 -pg -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_PG_NO_SUPPORT %s < %t.log
// LINK_PG_NO_SUPPORT: error: the clang compiler does not support -pg option on Darwin

// Check that clang links with libgcc_s.1 for iOS 4 and earlier, but not arm64.
// RUN: %clang -target armv7-apple-ios4.0 -miphoneos-version-min=4.0 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_IOS_LIBGCC_S %s < %t.log
// LINK_IOS_LIBGCC_S: lgcc_s.1

// RUN: %clang -target arm64-apple-ios4.0 -miphoneos-version-min=4.0 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_NO_IOS_ARM64_LIBGCC_S %s < %t.log
// LINK_NO_IOS_ARM64_LIBGCC_S-NOT: lgcc_s.1

// RUN: %clang -target x86_64-apple-darwin12 -rdynamic -### %t.o \
// RUN:   -fuse-ld= -mlinker-version=100 2> %t.log
// RUN: FileCheck -check-prefix=LINK_NO_EXPORT_DYNAMIC %s < %t.log
// LINK_NO_EXPORT_DYNAMIC: {{ld(.exe)?"}}
// LINK_NO_EXPORT_DYNAMIC-NOT: "-export_dynamic"

// RUN: %clang -target x86_64-apple-darwin12 -rdynamic -### %t.o \
// RUN:   -fuse-ld= -mlinker-version=137 2> %t.log
// RUN: FileCheck -check-prefix=LINK_EXPORT_DYNAMIC %s < %t.log
// LINK_EXPORT_DYNAMIC: {{ld(.exe)?"}}
// LINK_EXPORT_DYNAMIC: "-export_dynamic"

// RUN: %clang -target x86_64h-apple-darwin -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_X86_64H_ARCH %s < %t.log
//
// LINK_X86_64H_ARCH: {{ld(.exe)?"}}
// LINK_X86_64H_ARCH: "x86_64h"

// RUN: %clang -target x86_64-apple-darwin -arch x86_64 -arch x86_64h -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_X86_64H_MULTIARCH %s < %t.log
//
// LINK_X86_64H_MULTIARCH: {{ld(.exe)?"}}
// LINK_X86_64H_MULTIARCH: "x86_64"
//
// LINK_X86_64H_MULTIARCH: {{ld(.exe)?"}}
// LINK_X86_64H_MULTIARCH: "x86_64h"

// Check for the linker options to specify the iOS version when the
// IPHONEOS_DEPLOYMENT_TARGET variable is used instead of the command-line
// deployment target options.
// RUN: env IPHONEOS_DEPLOYMENT_TARGET=7.0 \
// RUN:   %clang -target arm64-apple-darwin -fuse-ld= -mlinker-version=400 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_IPHONEOS_VERSION_MIN %s < %t.log
// RUN: env IPHONEOS_DEPLOYMENT_TARGET=7.0 \
// RUN:   %clang -target i386-apple-darwin -fuse-ld= -mlinker-version=400 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_IOS_SIMULATOR_VERSION_MIN %s < %t.log
// LINK_IPHONEOS_VERSION_MIN: -iphoneos_version_min
// LINK_IOS_SIMULATOR_VERSION_MIN: -ios_simulator_version_min

// Ditto for tvOS....
// RUN: env TVOS_DEPLOYMENT_TARGET=7.0 \
// RUN:   %clang -target armv7-apple-darwin -fuse-ld= -mlinker-version=400 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_TVOS_VERSION_MIN %s < %t.log
// RUN: env TVOS_DEPLOYMENT_TARGET=7.0 \
// RUN:   %clang -target x86_64-apple-darwin -fuse-ld= -mlinker-version=400 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_TVOS_SIMULATOR_VERSION_MIN %s < %t.log
// LINK_TVOS_VERSION_MIN: -tvos_version_min
// LINK_TVOS_SIMULATOR_VERSION_MIN: -tvos_simulator_version_min

// ...and for watchOS.
// RUN: env WATCHOS_DEPLOYMENT_TARGET=2.0 \
// RUN:   %clang -target armv7k-apple-darwin -fuse-ld= -mlinker-version=400 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_WATCHOS_VERSION_MIN %s < %t.log
// RUN: env WATCHOS_DEPLOYMENT_TARGET=2.0 \
// RUN:   %clang -target i386-apple-darwin -fuse-ld= -mlinker-version=400 -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=LINK_WATCHOS_SIMULATOR_VERSION_MIN %s < %t.log
// LINK_WATCHOS_VERSION_MIN: -watchos_version_min
// LINK_WATCHOS_SIMULATOR_VERSION_MIN: -watchos_simulator_version_min

// Check -iframework gets forward to ld as -F
// RUN: %clang -target x86_64-apple-darwin %s -iframework Bar -framework Foo -### 2>&1 | \
// RUN:   FileCheck --check-prefix=LINK-IFRAMEWORK %s
// LINK-IFRAMEWORK: {{ld(.exe)?"}}
// LINK-IFRAMEWORK: "-FBar"

// Check ld64 accepts up to 5 digits with no extra characters
// RUN: %clang -target x86_64-apple-darwin12 %s -### -o %t \
// RUN:   -fuse-ld= -mlinker-version=133.3 2> %t.log
// RUN: %clang -target x86_64-apple-darwin12 %s -### -o %t \
// RUN:   -fuse-ld= -mlinker-version=133.3.0 2>> %t.log
// RUN: %clang -target x86_64-apple-darwin12 %s -### -o %t \
// RUN:   -fuse-ld= -mlinker-version=133.3.0.1 2>> %t.log
// RUN: %clang -target x86_64-apple-darwin12 %s -### -o %t \
// RUN:   -fuse-ld= -mlinker-version=133.3.0.1.2 2>> %t.log
// RUN: %clang -target x86_64-apple-darwin12 %s -### -o %t \
// RUN:   -fuse-ld= -mlinker-version=133.3.0.1.2.6 2>> %t.log
// RUN: %clang -target x86_64-apple-darwin12 %s -### -o %t \
// RUN:   -fuse-ld= -mlinker-version=133.3.0.1.a 2>> %t.log
// RUN: %clang -target x86_64-apple-darwin12 %s -### -o %t \
// RUN:   -fuse-ld= -mlinker-version=133.3.0.1a 2>> %t.log
// RUN: FileCheck -check-prefix=LINK_VERSION_DIGITS %s < %t.log
// LINK_VERSION_DIGITS-NOT: invalid version number in '-mlinker-version=133.3'
// LINK_VERSION_DIGITS-NOT: invalid version number in '-mlinker-version=133.3.0'
// LINK_VERSION_DIGITS-NOT: invalid version number in '-mlinker-version=133.3.0.1'
// LINK_VERSION_DIGITS: invalid version number in '-mlinker-version=133.3.0.1.2'
// LINK_VERSION_DIGITS: invalid version number in '-mlinker-version=133.3.0.1.2.6'
// LINK_VERSION_DIGITS: invalid version number in '-mlinker-version=133.3.0.1.a'
// LINK_VERSION_DIGITS: invalid version number in '-mlinker-version=133.3.0.1a'

// RUN: %clang -target x86_64-apple-darwin12 -fprofile-instr-generate -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=PROFILE_SECTALIGN %s < %t.log
// RUN: %clang -target arm64-apple-ios12 -fprofile-instr-generate -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=PROFILE_SECTALIGN %s < %t.log
// PROFILE_SECTALIGN: "-sectalign" "__DATA" "__llvm_prf_cnts" "0x4000" "-sectalign" "__DATA" "__llvm_prf_data" "0x4000"

// RUN: %clang -target x86_64-apple-darwin12 -fprofile-instr-generate -exported_symbols_list /dev/null -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=PROFILE_EXPORT %s < %t.log
// RUN: %clang -target x86_64-apple-darwin12 -fprofile-instr-generate -Wl,-exported_symbols_list,/dev/null -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=PROFILE_EXPORT %s < %t.log
// RUN: %clang -target x86_64-apple-darwin12 -fprofile-instr-generate -Wl,-exported_symbol,foo -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=PROFILE_EXPORT %s < %t.log
// RUN: %clang -target x86_64-apple-darwin12 -fprofile-instr-generate -Xlinker -exported_symbol -Xlinker foo -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=PROFILE_EXPORT %s < %t.log
// RUN: %clang -target x86_64-apple-darwin12 -fprofile-instr-generate -Xlinker -exported_symbols_list -Xlinker /dev/null -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=PROFILE_EXPORT %s < %t.log
// PROFILE_EXPORT: "-exported_symbol" "___llvm_profile_filename" "-exported_symbol" "___llvm_profile_raw_version"
//
// RUN: %clang -target x86_64-apple-darwin12 -fprofile-instr-generate --coverage -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=NO_PROFILE_EXPORT %s < %t.log
// NO_PROFILE_EXPORT-NOT: "-exported_symbol"
//
// RUN: %clang -target x86_64-apple-darwin12 --coverage -exported_symbols_list /dev/null -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=GCOV_EXPORT %s < %t.log
// RUN: %clang -target x86_64-apple-darwin12 -fprofile-arcs -Wl,-exported_symbols_list,/dev/null -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=GCOV_EXPORT %s < %t.log
// RUN: %clang -target x86_64-apple-darwin12 -fprofile-arcs -Wl,-exported_symbol,foo -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=GCOV_EXPORT %s < %t.log
// RUN: %clang -target x86_64-apple-darwin12 -fprofile-arcs -Xlinker -exported_symbol -Xlinker foo -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=GCOV_EXPORT %s < %t.log
// RUN: %clang -target x86_64-apple-darwin12 -fprofile-arcs -Xlinker -exported_symbols_list -Xlinker /dev/null -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=GCOV_EXPORT %s < %t.log
// GCOV_EXPORT: "-exported_symbol" "___gcov_dump"
// GCOV_EXPORT: "-exported_symbol" "___gcov_reset"
//
// Check that we can pass the outliner down to the linker.
// RUN: env IPHONEOS_DEPLOYMENT_TARGET=7.0 \
// RUN:   %clang -target arm64-apple-darwin -moutline -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=MOUTLINE %s < %t.log
// MOUTLINE: {{ld(.exe)?"}}
// MOUTLINE-SAME: "-mllvm" "-enable-machine-outliner" "-mllvm" "-enable-linkonceodr-outlining"
// RUN: env IPHONEOS_DEPLOYMENT_TARGET=7.0 \
// RUN:   %clang -target arm64-apple-darwin -mno-outline -### %t.o 2> %t.log
// RUN: FileCheck -check-prefix=MNO_OUTLINE %s < %t.log
// MNO_OUTLINE: {{ld(.exe)?"}}
// MNO_OUTLINE-SAME: "-mllvm" "-enable-machine-outliner=never"
