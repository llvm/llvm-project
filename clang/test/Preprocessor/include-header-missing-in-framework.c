// RUN: %clang_cc1 -fsyntax-only -F %S/Inputs -verify %s
// RUN: %clang_cc1 -fsyntax-only -F %S/Inputs -DTYPO_CORRECTION -verify %s

// After finding a requested framework, we don't look for the same framework in
// a different location even if requested header is not found in the framework.
// It can be confusing when there is a framework with required header later in
// header search paths. Mention in diagnostics where the header lookup stopped.

#ifndef TYPO_CORRECTION
#include <TestFramework/NotExistingHeader.h>
// expected-error@-1 {{'TestFramework/NotExistingHeader.h' file not found}}
// expected-note@-2 {{did not find header 'NotExistingHeader.h' in framework 'TestFramework' (loaded from}}

#else
// Don't emit extra note for unsuccessfully typo-corrected include.
#include <#TestFramework/NotExistingHeader.h>
// expected-error@-1 {{'#TestFramework/NotExistingHeader.h' file not found}}
#endif // TYPO_CORRECTION
