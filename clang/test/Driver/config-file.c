//--- Config file search directories
//
// RUN: %clang --config-system-dir=%S/Inputs/config --config-user-dir=%S/Inputs/config2 -o /dev/null -v 2>&1 | FileCheck %s -check-prefix CHECK-DIRS
// CHECK-DIRS: System configuration file directory: {{.*}}/Inputs/config
// CHECK-DIRS: User configuration file directory: {{.*}}/Inputs/config2


//--- Config file (full path) in output of -###
//
// RUN: %clang --config %S/Inputs/config-1.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-HHH
// RUN: %clang --config=%S/Inputs/config-1.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-HHH
// CHECK-HHH: Configuration file: {{.*}}Inputs{{.}}config-1.cfg
// CHECK-HHH: -Werror
// CHECK-HHH: -std=c99


//--- Config file (full path) in output of -v
//
// RUN: %clang --config %S/Inputs/config-1.cfg -S %s -o /dev/null -v 2>&1 | FileCheck %s -check-prefix CHECK-V
// CHECK-V: Configuration file: {{.*}}Inputs{{.}}config-1.cfg
// CHECK-V: -Werror
// CHECK-V: -std=c99


//--- Config file in output of -###
//
// RUN: %clang --config-system-dir=%S/Inputs --config-user-dir= --config config-1.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-HHH2
// CHECK-HHH2: Configuration file: {{.*}}Inputs{{.}}config-1.cfg
// CHECK-HHH2: -Werror
// CHECK-HHH2: -std=c99


//--- Config file in output of -v
//
// RUN: %clang --config-system-dir=%S/Inputs --config-user-dir= --config config-1.cfg -S %s -o /dev/null -v 2>&1 | FileCheck %s -check-prefix CHECK-V2
// CHECK-V2: Configuration file: {{.*}}Inputs{{.}}config-1.cfg
// CHECK-V2: -Werror
// CHECK-V2: -std=c99


//--- Nested config files
//
// RUN: %clang --config-system-dir=%S/Inputs --config-user-dir= --config config-2.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NESTED
// CHECK-NESTED: Configuration file: {{.*}}Inputs{{.}}config-2.cfg
// CHECK-NESTED: -Wundefined-func-template

// RUN: %clang --config-system-dir=%S/Inputs --config-user-dir=%S/Inputs/config --config config-6.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NESTED2
// CHECK-NESTED2: Configuration file: {{.*}}Inputs{{.}}config-6.cfg
// CHECK-NESTED2: -isysroot
// CHECK-NESTED2-SAME: /opt/data


// RUN: %clang --config %S/Inputs/config-2a.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NESTEDa
// CHECK-NESTEDa: Configuration file: {{.*}}Inputs{{.}}config-2a.cfg
// CHECK-NESTEDa: -isysroot
// CHECK-NESTEDa-SAME: /opt/data

// RUN: %clang --config-system-dir=%S/Inputs --config-user-dir= --config config-2a.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NESTED2a
// CHECK-NESTED2a: Configuration file: {{.*}}Inputs{{.}}config-2a.cfg
// CHECK-NESTED2a: -isysroot
// CHECK-NESTED2a-SAME: /opt/data


//--- Unused options in config file do not produce warnings
//
// RUN: %clang --config %S/Inputs/config-4.cfg -S %s -o /dev/null -v 2>&1 | FileCheck %s -check-prefix CHECK-UNUSED
// CHECK-UNUSED-NOT: argument unused during compilation:
// CHECK-UNUSED-NOT: 'linker' input unused


//--- User directory is searched first.
//
// RUN: %clang --config-system-dir=%S/Inputs/config --config-user-dir=%S/Inputs/config2 --config config-4.cfg -S %s -o /dev/null -v 2>&1 | FileCheck %s -check-prefix CHECK-PRECEDENCE
// CHECK-PRECEDENCE: Configuration file: {{.*}}Inputs{{.}}config2{{.}}config-4.cfg
// CHECK-PRECEDENCE: -Wall


//--- Multiple configuration files can be specified.
// RUN: %clang --config-system-dir=%S/Inputs/config --config-user-dir= --config config-4.cfg --config %S/Inputs/config2/config-4.cfg -S %s -o /dev/null -v 2>&1 | FileCheck %s -check-prefix CHECK-TWO-CONFIGS
// CHECK-TWO-CONFIGS: Configuration file: {{.*}}Inputs{{.}}config{{.}}config-4.cfg
// CHECK-TWO-CONFIGS-NEXT: Configuration file: {{.*}}Inputs{{.}}config2{{.}}config-4.cfg
// CHECK-TWO-CONFIGS: -isysroot
// CHECK-TWO-CONFIGS-SAME: /opt/data
// CHECK-TWO-CONFIGS-SAME: -Wall

//--- The linker input flags should be moved to the end of input list and appear only when linking.
// RUN: %clang --target=aarch64-unknown-linux-gnu --config %S/Inputs/config-l.cfg %s -lmylib -Wl,foo.a -### 2>&1 | FileCheck %s -check-prefix CHECK-LINKING
// RUN: %clang --target=aarch64-unknown-linux-gnu --config %S/Inputs/config-l.cfg -fopenmp=libomp %s -lmylib -Wl,foo.a -### 2>&1 | FileCheck %s -check-prefix CHECK-LINKING-LIBOMP-GOES-AFTER
// RUN: %clang --target=aarch64-unknown-linux-gnu --config %S/Inputs/config-l.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NOLINKING
// RUN: %clang --target=aarch64-unknown-linux-gnu --config %S/Inputs/config-l.cfg -fopenmp=libomp -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NOLINKING-OPENMP
// RUN: %clang --target=x86_64-pc-windows-msvc    --config %S/Inputs/config-l.cfg %s -lmylib -Wl,foo.lib -### 2>&1 | FileCheck %s -check-prefix CHECK-LINKING-MSVC
// RUN: %clang --target=x86_64-pc-windows-msvc    --config %S/Inputs/config-l.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NOLINKING-MSVC
// CHECK-LINKING: Configuration file: {{.*}}Inputs{{.}}config-l.cfg
// CHECK-LINKING: "-Wall"
// CHECK-LINKING: "--as-needed" "{{.*}}-{{.*}}.o" "-lmylib" "foo.a" "-lm" "-Bstatic" "-lhappy" "-Bdynamic"
// CHECK-LINKING-LIBOMP-GOES-AFTER: Configuration file: {{.*}}Inputs{{.}}config-l.cfg
// CHECK-LINKING-LIBOMP-GOES-AFTER: "-Wall" {{.*}}"-fopenmp"
// CHECK-LINKING-LIBOMP-GOES-AFTER: "--as-needed" "{{.*}}-{{.*}}.o" "-lmylib" "foo.a" "-lm" "-Bstatic" "-lhappy" "-Bdynamic" {{.*}}"-lomp"
// CHECK-NOLINKING: Configuration file: {{.*}}Inputs{{.}}config-l.cfg
// CHECK-NOLINKING: "-Wall"
// CHECK-NOLINKING-NO: "-lm" "-Bstatic" "-lhappy" "-Bdynamic"
// CHECK-NOLINKING-OPENMP: Configuration file: {{.*}}Inputs{{.}}config-l.cfg
// CHECK-NOLINKING-OPENMP: "-Wall" {{.*}}"-fopenmp"
// CHECK-NOLINKING-OPENMP-NO: "-lm" "-Bstatic" "-lhappy" "-Bdynamic" {{.*}}"-lomp"
// CHECK-LINKING-MSVC: Configuration file: {{.*}}Inputs{{.}}config-l.cfg
// CHECK-LINKING-MSVC: "-Wall"
// CHECK-LINKING-MSVC: "--as-needed" "{{.*}}-{{.*}}.o" "mylib.lib" "foo.lib" "m.lib" "-Bstatic" "happy.lib" "-Bdynamic"
// CHECK-NOLINKING-MSVC: Configuration file: {{.*}}Inputs{{.}}config-l.cfg
// CHECK-NOLINKING-MSVC: "-Wall"
// CHECK-NOLINKING-MSVC-NO: "m.lib" "-Bstatic" "happy.lib" "-Bdynamic"
