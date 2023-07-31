// RUN: %clang -### -coverage -working-directory /no/such/dir/ input 2>&1 | FileCheck %s
// RUN: %clang -### -coverage -working-directory %p/Inputs no_such_file.cpp -c 2>&1 | FileCheck %s --check-prefix=CHECK_NO_FILE
// RUN: %clang -### -coverage -working-directory %p/Inputs pchfile.cpp -c 2>&1 | FileCheck %s --check-prefix=CHECK_WORKS

// CHECK: unable to set working directory: /no/such/dir/

// CHECK_NO_FILE: no such file or directory: 'no_such_file.cpp'

// CHECK_WORKS: "-coverage-notes-file" "{{[^"]+}}test{{/|\\\\}}Driver{{/|\\\\}}Inputs{{/|\\\\}}pchfile.gcno"
// CHECK_WORKS: "-working-directory" "{{[^"]+}}test{{/|\\\\}}Driver{{/|\\\\}}Inputs"
// CHECK_WORKS: "-fdebug-compilation-dir={{[^"]+}}test{{/|\\\\}}Driver{{/|\\\\}}Inputs"
