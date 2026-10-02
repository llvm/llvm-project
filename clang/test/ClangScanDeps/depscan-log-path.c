// Diagnostics for the -fdepscan-log-path driver flag during dependency scanning:
// an empty value is rejected, an inconsistent value across commands in one scan
// is rejected, and a consistent value is accepted (and aggregated).

// UNSUPPORTED: system-windows
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/empty.json.template        > %t/empty.json
// RUN: sed -e "s|DIR|%/t|g" %t/inconsistent.json.template > %t/inconsistent.json
// RUN: sed -e "s|DIR|%/t|g" %t/inconsistent2.json.template > %t/inconsistent2.json
// RUN: sed -e "s|DIR|%/t|g" %t/consistent.json.template   > %t/consistent.json

// An explicitly empty value is rejected.
// RUN: not clang-scan-deps -compilation-database %t/empty.json \
// RUN:   -format experimental-full -j 1 2>&1 | FileCheck %s --check-prefix=EMPTY
// EMPTY: error: '-fdepscan-log-path=' requires a non-empty file path

// Different log paths across commands in one scan are rejected.
// RUN: not clang-scan-deps -compilation-database %t/inconsistent.json \
// RUN:   -format experimental-full -j 1 2>&1 | FileCheck %s --check-prefix=CONFLICT

// One command with a valid flag, the other does not have a flag in effect.
// RUN: not clang-scan-deps -compilation-database %t/inconsistent2.json \
// RUN:   -format experimental-full -j 1 2>&1 | FileCheck %s --check-prefix=CONFLICT
// CONFLICT: error: '-fdepscan-log-path' set inconsistently within a dependency scan

// The same log path across commands is fine; both are aggregated into one log.
// RUN: clang-scan-deps -compilation-database %t/consistent.json \
// RUN:   -format experimental-full -j 1 -o %t/deps.json
// RUN: FileCheck %s --check-prefix=OK --input-file %t/scan.log
// OK: logging_start
// OK: starting scanning command:{{.*}}tu.c
// OK: starting scanning command:{{.*}}tu2.c
// OK: logging_end


//--- empty.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fdepscan-log-path=",
  "file": "DIR/tu.c"
}]

//--- inconsistent.json.template
[
{ "directory": "DIR", "command": "clang -fsyntax-only DIR/tu.c -fdepscan-log-path=DIR/a.log",  "file": "DIR/tu.c" },
{ "directory": "DIR", "command": "clang -fsyntax-only DIR/tu2.c -fdepscan-log-path=DIR/b.log", "file": "DIR/tu2.c" }
]

//--- consistent.json.template
[
{ "directory": "DIR", "command": "clang -fsyntax-only DIR/tu.c -fdepscan-log-path=DIR/scan.log",  "file": "DIR/tu.c" },
{ "directory": "DIR", "command": "clang -fsyntax-only DIR/tu2.c -fdepscan-log-path=DIR/scan.log", "file": "DIR/tu2.c" }
]

//--- inconsistent2.json.template
[
{ "directory": "DIR", "command": "clang -fsyntax-only DIR/tu.c",  "file": "DIR/tu.c" },
{ "directory": "DIR", "command": "clang -fsyntax-only DIR/tu2.c -fdepscan-log-path=DIR/b.log", "file": "DIR/tu2.c" }
]

//--- tu.c
void foo(void) {}
//--- tu2.c
void bar(void) {}
