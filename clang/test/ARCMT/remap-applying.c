a bc

// RUN: echo "[{\"file\": \"%/s\", \"offset\": 1, \"remove\": 2, }]" > %t.remap
// RUN: c-arcmt-test %t.remap | arcmt-test -verify-transformed-files %s.result
