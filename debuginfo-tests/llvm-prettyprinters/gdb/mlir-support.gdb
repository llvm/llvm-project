# RUN: gdb -q -batch -n -iex 'source %mlir_src_root/utils/gdb-scripts/prettyprinters.py' -x %s %llvm_tools_dir/check-gdb-mlir-support | FileCheck %s --dump-input=fail
# REQUIRES: debug-info

break main
run

# CHECK: "foo"
p Identifier
