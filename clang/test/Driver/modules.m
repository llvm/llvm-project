// RUN: %clang -fmodules -fno-modules -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MODULES %s
// CHECK-NO-MODULES-NOT: -fmodules

// RUN: %clang -fmodules -fno-modules -fmodules -### %s 2>&1 | FileCheck -check-prefix=CHECK-HAS-MODULES %s
// CHECK-HAS-MODULES: -fmodules

// RUN: not %clang -fmodules -fbuild-session-file=doesntexist -### %s 2>&1 | FileCheck -check-prefix=NOFILE %s
// NOFILE: no such file or directory: 'doesntexist'

// RUN: touch -m -a -t 201008011501 %t.build-session-file
// RUN: %clang -fmodules -fbuild-session-file=%t.build-session-file -### %s 2>&1 | FileCheck -check-prefix=TIMESTAMP_ONLY %s

// RUN: %clang -fmodules -fbuild-session-timestamp=1280703457 -### %s 2>&1 | FileCheck -check-prefix=TIMESTAMP_ONLY %s
// TIMESTAMP_ONLY: -fbuild-session-timestamp=128{{([[:digit:]]{7})[^[:digit:]]}}

// RUN: not %clang -fmodules -fbuild-session-file=%t.build-session-file -fbuild-session-timestamp=123 -### %s 2>&1 | FileCheck -check-prefix=CONFLICT %s
// CONFLICT: error: invalid argument '-fbuild-session-file={{.*}}.build-session-file' not allowed with '-fbuild-session-timestamp'

// RUN: %clang -fmodules -fbuild-session-timestamp=123 -fmodules-validate-once-per-build-session -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_ONCE %s
// MODULES_VALIDATE_ONCE: -fbuild-session-timestamp=123
// MODULES_VALIDATE_ONCE: -fmodules-validate-once-per-build-session

// RUN: %clang -fmodules -fbuild-session-file=%t.build-session-file -fmodules-validate-once-per-build-session -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_ONCE_FILE %s
// MODULES_VALIDATE_ONCE_FILE: -fbuild-session-timestamp=128{{([[:digit:]]{7})[^[:digit:]]}}
// MODULES_VALIDATE_ONCE_FILE: -fmodules-validate-once-per-build-session

// RUN: not %clang -fmodules -fmodules-validate-once-per-build-session -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_ONCE_ERR %s
// MODULES_VALIDATE_ONCE_ERR: option '-fmodules-validate-once-per-build-session' requires '-fbuild-session-timestamp=<seconds since Epoch>' or '-fbuild-session-file=<file>'

// RUN: %clang -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_SYSTEM_HEADERS_DEFAULT %s
// MODULES_VALIDATE_SYSTEM_HEADERS_DEFAULT-NOT: -fmodules-validate-system-headers

// RUN: %clang -fmodules -fsyntax-only -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_SYSTEM_HEADERS_DEFAULT_MOD %s
// MODULES_VALIDATE_SYSTEM_HEADERS_DEFAULT_MOD: -fmodules-validate-system-headers

// RUN: %clang -fmodules -fmodules-validate-system-headers -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_SYSTEM_HEADERS %s
// MODULES_VALIDATE_SYSTEM_HEADERS: -fmodules-validate-system-headers

// RUN: %clang -fmodules -fno-modules-validate-system-headers -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_SYSTEM_HEADERS_NOSYSVALID %s
// MODULES_VALIDATE_SYSTEM_HEADERS_NOSYSVALID-NOT: -fmodules-validate-system-headers

// RUN: %clang -### %s 2>&1 | FileCheck -check-prefix=MODULES_DISABLE_DIAGNOSTIC_VALIDATION_DEFAULT %s
// MODULES_DISABLE_DIAGNOSTIC_VALIDATION_DEFAULT-NOT: -fmodules-disable-diagnostic-validation

// RUN: %clang -fmodules -fmodules-disable-diagnostic-validation -### %s 2>&1 | FileCheck -check-prefix=MODULES_DISABLE_DIAGNOSTIC_VALIDATION %s
// MODULES_DISABLE_DIAGNOSTIC_VALIDATION: -fmodules-disable-diagnostic-validation

// RUN: %clang -fmodules -### %s 2>&1 | FileCheck -check-prefix=MODULES_PREBUILT_PATH_DEFAULT %s
// MODULES_PREBUILT_PATH_DEFAULT-NOT: -fprebuilt-module-path

// RUN: %clang -fmodules -fprebuilt-module-path=foo -fprebuilt-module-path=bar -### %s 2>&1 | FileCheck -check-prefix=MODULES_PREBUILT_PATH %s
// MODULES_PREBUILT_PATH: "-fprebuilt-module-path=foo"
// MODULES_PREBUILT_PATH: "-fprebuilt-module-path=bar"

// RUN: %clang -fmodules -fmodule-map-file=foo.map -fmodule-map-file=bar.map -### %s 2>&1 | FileCheck -check-prefix=CHECK-MODULE-MAP-FILES %s
// CHECK-MODULE-MAP-FILES: "-fmodules"
// CHECK-MODULE-MAP-FILES: "-fmodule-map-file=foo.map"
// CHECK-MODULE-MAP-FILES: "-fmodule-map-file=bar.map"

// RUN: %clang -fmodules -fbuiltin-module-map -### %s 2>&1 | FileCheck -check-prefix=CHECK-BUILTIN-MODULE-MAP %s
// CHECK-BUILTIN-MODULE-MAP: "-fmodules"
// CHECK-BUILTIN-MODULE-MAP: "-fmodule-map-file={{.*}}include{{/|\\\\}}module.modulemap"

// RUN: %clang -fmodules -fmodule-file=foo.pcm -fmodule-file=bar.pcm -### %s 2>&1 | FileCheck -check-prefix=CHECK-MODULE-FILES %s
// CHECK-MODULE-FILES: "-fmodules"
// CHECK-MODULE-FILES: "-fmodule-file=foo.pcm"
// CHECK-MODULE-FILES: "-fmodule-file=bar.pcm"

// RUN: %clang -fno-modules -fmodule-file=foo.pcm -fmodule-file=bar.pcm -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MODULE-FILES %s
// CHECK-NO-MODULE-FILES-NOT: "-fmodules"
// CHECK-NO-MODULE-FILES-NOT: "-fmodule-file=foo.pcm"
// CHECK-NO-MODULE-FILES-NOT: "-fmodule-file=bar.pcm"

// RUN: %clang -fno-modules -fbuild-session-file=%t.build-session-file -### %s 2>&1 | FileCheck -check-prefix=SESSION_FLAG %s
// RUN: %clang -fno-modules -fbuild-session-timestamp=123 -### %s 2>&1 | FileCheck -check-prefix=SESSION_FLAG %s
// SESSION_FLAG-NOT: -fbuild-session-timestamp

// RUN: %clang -fno-modules -fmodules-validate-once-per-build-session -### %s 2>&1 | FileCheck -check-prefix=VALIDATE_ONCE_FLAG %s
// VALIDATE_ONCE_FLAG-NOT: -fmodules-validate-once-per-build-session

// RUN: %clang -fno-modules -fmodules-validate-system-headers -### %s 2>&1 | FileCheck -check-prefix=VALIDATE_SYSTEM_FLAG %s
// VALIDATE_SYSTEM_FLAG-NOT: -fmodules-validate-system-headers
