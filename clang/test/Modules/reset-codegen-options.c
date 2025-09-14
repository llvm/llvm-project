// REQUIRES: modules

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fmodules -emit-module-interface -fmodule-name=TestModule -o - %s \
// RUN:   -fmodule-file=module.pcm 2>&1 | FileCheck %s --implicit-check-not="MainFileName" --implicit-check-not="DebugCompilationDir"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-pch -o - %s 2>&1 | FileCheck %s --implicit-check-not="DwarfDebugFlags"

// CHECK-NOT: MainFileName
// CHECK-NOT: DwarfDebugFlags
// CHECK-NOT: DebugCompilationDir
// CHECK-NOT: CoverageCompilationDir
// CHECK-NOT: CoverageDataFile
// CHECK-NOT: CoverageNotesFile
// CHECK-NOT: ProfileInstrumentUsePath
// CHECK-NOT: SampleProfileFile
// CHECK-NOT: ProfileRemappingFile

int foo() {
    return 42;
  }
  