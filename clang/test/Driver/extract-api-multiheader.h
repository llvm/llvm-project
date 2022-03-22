// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang -target x86_64-unknown-unknown -ccc-print-phases -extract-api %t/first-header.h %t/second-header.h 2> %t1
// RUN: echo 'END' >> %t1
// RUN: FileCheck -check-prefix EXTRACT-API-PHASES -input-file %t1 %s

// EXTRACT-API-PHASES: 0: input
// EXTRACT-API-PHASES-SAME: , c-header
// EXTRACT-API-PHASES-NEXT: 1: preprocessor, {0}, c-header-cpp-output
// EXTRACT-API-PHASES-NEXT: 2: input
// EXTRACT-API-PHASES-SAME: , c-header
// EXTRACT-API-PHASES-NEXT: 3: preprocessor, {2}, c-header-cpp-output
// EXTRACT-API-PHASES-NEXT: 4: api-extractor, {1, 3}, api-information
// EXTRACT-API-PHASES-NOT: 5:
// EXTRACT-API-PHASES: END

//--- first-header.h

void dummy_function(void);

//--- second-header.h

void other_dummy_function(void);
