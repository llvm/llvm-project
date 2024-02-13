// Check non-darwin triple is rejected.
// RUN: not %clang -target x86_64-unknown-unknown -installapi %s 2> %t 
// RUN: FileCheck --check-prefix INVALID_INSTALLAPI -input-file %t %s

// INVALID_INSTALLAPI: error: InstallAPI is not supported for 'x86_64-unknown-unknown'

// Check installapi phases.
// RUN: %clang -target x86_64-apple-macos11 -ccc-print-phases -installapi %s 2> %t 
// RUN: FileCheck --check-prefix INSTALLAPI_PHASES -input-file %t %s

// INSTALLAPI_PHASES: 0: input,
// INSTALLAPI_PHASES: installapi,
// INSTALLAPI_PHASES-SAME: tbd 
