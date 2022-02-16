// RUN: env RC_DEBUG_PREFIX_MAP=old=new \
// RUN:  %clang -target arm64-apple-darwin -### -c -g %s 2>&1 | FileCheck %s
// RUN: env RC_DEBUG_PREFIX_MAP=illegal \
// RUN:  %clang -target arm64-apple-darwin -### -c -g %s 2>&1 | FileCheck %s --check-prefix=ERR
// CHECK: "-fdebug-prefix-map=old=new" 
// ERR: invalid argument 'illegal'
