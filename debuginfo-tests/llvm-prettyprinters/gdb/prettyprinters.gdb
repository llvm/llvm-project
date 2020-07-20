# RUN: gdb -q -batch -n -iex 'source %llvm_src_root/utils/gdb-scripts/prettyprinters.py' -x %s %llvm_tools_dir/prettyprinters | FileCheck %s

break main
run

# CHECK: llvm::ArrayRef of length 3 = {1, 2, 3}
p ArrayRef

# CHECK: llvm::ArrayRef of length 3 = {1, 2, 3}
p MutableArrayRef

# CHECK: llvm::DenseMap with 2 elements = {
# CHECK:   [4] = 5,
# CHECK:   [6] = 7,
# CHECK: }
p DenseMap

# CHECK: llvm::Expected = {value = 8}
p ExpectedValue

# CHECK: llvm::Expected is error
p ExpectedError

# CHECK: llvm::Optional = {value = 9}
p OptionalValue

# CHECK: llvm::Optional is not initialized
p OptionalNone

# CHECK: llvm::SmallVector of Size 3, Capacity 5 = {10, 11, 12}
p SmallVector

# CHECK: "foo"
p SmallString

# CHECK: "bar"
p StringRef

# CHECK: "\"foo\"\"bar\""
p Twine

