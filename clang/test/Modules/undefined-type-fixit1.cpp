// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -fmodules-search-all -I %S/Inputs/undefined-type-fixit %s -verify

//#include "public1.h"
#include "public2.h"
#include "public2sub.h"

use_this1 client_variable1; // expected-error{{'use_this1' must be declared}}
use_this2 client_variable2;
use_this2sub client_variable2sub;

// expected-note@Inputs/undefined-type-fixit/public1.h:4 {{declaration here is not visible}}
