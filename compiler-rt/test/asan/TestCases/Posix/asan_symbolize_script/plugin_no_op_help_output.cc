// Check help output.
// RUN: %asan_symbolize --log-level info --plugins %S/plugin_no_op.py --help 2>&1 | FileCheck %s
// CHECK: Registering plugin NoOpPlugin
// CHECK: Adding --unlikely-option-name-XXX option
// CHECK: optional arguments:
// CHECK: --unlikely-option-name-XXX

