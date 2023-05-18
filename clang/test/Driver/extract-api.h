// RUN: %clang -target x86_64-unknown-unknown -ccc-print-phases -extract-api %s 2> %t
// RUN: echo 'END' >> %t
// RUN: FileCheck -check-prefix EXTRACT-API-PHASES -input-file %t %s

// EXTRACT-API-PHASES: 0: input,
// EXTRACT-API-PHASES-SAME: , c-header
// EXTRACT-API-PHASES-NEXT: 1: preprocessor, {0}, c-header-cpp-output
// EXTRACT-API-PHASES-NEXT: 2: api-extractor, {1}, api-information
// EXTRACT-API-PHASES-NOT: 3:
// EXTRACT-API-PHASES: END
