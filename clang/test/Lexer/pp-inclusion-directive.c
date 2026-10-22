// RUN: %clang_cc1 -fsyntax-only -detailed-preprocessing-record -I %S -I %S/Inputs %s
// Test that PreprocessingRecord::InclusionDirective correctly handles various
// inclusion directive kinds (include, import, include_next).

#include "pp-inclusion-directive.h"
#include <pp-inclusion-directive.h>
#import "pp-inclusion-directive.h"
#include_next "pp-inclusion-directive.h"

