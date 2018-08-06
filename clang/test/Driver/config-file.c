//--- Config file search directories
//
// RUN: %clang --config-system-dir=%S/Inputs/config --config-user-dir=%S/Inputs/config2 -o /dev/null -v 2>&1 | FileCheck %s -check-prefix CHECK-DIRS
// CHECK-DIRS: System configuration file directory: {{.*}}/Inputs/config
// CHECK-DIRS: User configuration file directory: {{.*}}/Inputs/config2


//--- Config file (full path) in output of -###
//
// RUN: %clang --config %S/Inputs/config-1.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-HHH
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

// RUN: %clang --config-system-dir=%S/Inputs --config-user-dir= --config config-2.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NESTED2
// CHECK-NESTED2: Configuration file: {{.*}}Inputs{{.}}config-2.cfg
// CHECK-NESTED2: -Wundefined-func-template


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
// RUN: %clang --config-system-dir=%S/Inputs/config --config-user-dir=%S/Inputs/config2 --config config-4 -S %s -o /dev/null -v 2>&1 | FileCheck %s -check-prefix CHECK-PRECEDENCE
// CHECK-PRECEDENCE: Configuration file: {{.*}}Inputs{{.}}config2{{.}}config-4.cfg
// CHECK-PRECEDENCE: -Wall
