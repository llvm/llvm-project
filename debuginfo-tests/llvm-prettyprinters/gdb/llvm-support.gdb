# RUN: gdb -q -batch -n -iex 'source %llvm_src_root/utils/gdb-scripts/prettyprinters.py' -x %s %llvm_tools_dir/check-gdb-llvm-support | FileCheck %s
# REQUIRES: debug-info

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

# CHECK: {pointer = 0xabc, value = 1}
p PointerIntPair

# CHECK: Containing int * = {pointer = 0xabc}
p PointerUnion

# CHECK: PointerUnionMembers<llvm::PointerUnion<Z*, float*>,
p RawPrintingPointerUnion

# Switch to print pretty adds newlines to the following statements.
set print pretty

# CHECK: {
# CHECK:   [0] = {
# CHECK:     <llvm::ilist_node<IlistNode, llvm::ilist_tag<A> >> = {
# CHECK:       prev = [[Ilist_Sentinel:0x.*]] <Ilist>,
# CHECK:       next = [[Node_14:0x.*]]
# CHECK:     },
# CHECK:     <llvm::ilist_node<IlistNode, llvm::ilist_tag<B> >> = {
# CHECK:       prev = [[Node_14]],
# CHECK:       next = [[SimpleIlist_Sentinel:0x.*]] <SimpleIlist>
# CHECK:     },
# CHECK:     members of IlistNode:
# CHECK:     Value = 13
# CHECK:   },
# CHECK:   [1] = {
# CHECK:     <llvm::ilist_node<IlistNode, llvm::ilist_tag<A> >> = {
# CHECK:       prev = [[Node_13:0x.*]],
# CHECK:       next = [[Node_15:0x.*]]
# CHECK:     },
# CHECK:     <llvm::ilist_node<IlistNode, llvm::ilist_tag<B> >> = {
# CHECK:       prev = [[Node_15]],
# CHECK:       next = [[Node_13]]
# CHECK:     },
# CHECK:     members of IlistNode:
# CHECK:     Value = 14
# CHECK:   },
# CHECK:   [2] = {
# CHECK:     <llvm::ilist_node<IlistNode, llvm::ilist_tag<A> >> = {
# CHECK:       prev = [[Node_14]],
# CHECK:       next = [[Ilist_Sentinel]] <Ilist>
# CHECK:     },
# CHECK:     <llvm::ilist_node<IlistNode, llvm::ilist_tag<B> >> = {
# CHECK:       prev = [[SimpleIlist_Sentinel]] <SimpleIlist>,
# CHECK:       next = [[Node_14]]
# CHECK:     },
# CHECK:     members of IlistNode:
# CHECK:     Value = 15
# CHECK:   }
# CHECK: }
p Ilist

# CHECK: {
# CHECK:   [0] = {
# CHECK:     <llvm::ilist_node<IlistNode, llvm::ilist_tag<A> >> = {
# CHECK:       prev = [[Node_14]],
# CHECK:       next = [[Ilist_Sentinel]] <Ilist>
# CHECK:     },
# CHECK:     <llvm::ilist_node<IlistNode, llvm::ilist_tag<B> >> = {
# CHECK:       prev = [[SimpleIlist_Sentinel]] <SimpleIlist>,
# CHECK:       next = [[Node_14]]
# CHECK:     },
# CHECK:     members of IlistNode:
# CHECK:     Value = 15
# CHECK:   },
# CHECK:   [1] = {
# CHECK:     <llvm::ilist_node<IlistNode, llvm::ilist_tag<A> >> = {
# CHECK:       prev = [[Node_13]],
# CHECK:       next = [[Node_15]]
# CHECK:     },
# CHECK:     <llvm::ilist_node<IlistNode, llvm::ilist_tag<B> >> = {
# CHECK:       prev = [[Node_15]],
# CHECK:       next = [[Node_13]]
# CHECK:     },
# CHECK:     members of IlistNode:
# CHECK:     Value = 14
# CHECK:   },
# CHECK:   [2] = {
# CHECK:     <llvm::ilist_node<IlistNode, llvm::ilist_tag<A> >> = {
# CHECK:       prev = [[Ilist_Sentinel]] <Ilist>,
# CHECK:       next = [[Node_14]]
# CHECK:     },
# CHECK:     <llvm::ilist_node<IlistNode, llvm::ilist_tag<B> >> = {
# CHECK:       prev = [[Node_14]],
# CHECK:       next = [[SimpleIlist_Sentinel]] <SimpleIlist>
# CHECK:     },
# CHECK:     members of IlistNode:
# CHECK:     Value = 13
# CHECK:   }
# CHECK: }
p SimpleIlist
