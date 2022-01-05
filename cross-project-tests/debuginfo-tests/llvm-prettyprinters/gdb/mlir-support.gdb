# RUN: gdb -q -batch -n \
# RUN:   -iex 'source %mlir_src_root/utils/gdb-scripts/prettyprinters.py' \
# RUN:   -iex 'source %llvm_src_root/utils/gdb-scripts/prettyprinters.py' \
# RUN:   -ex 'source -v %s' %llvm_tools_dir/check-gdb-mlir-support \
# RUN: | FileCheck %s
# REQUIRES: debug-info
# REQUIRES: mlir

break main
run
set print pretty on

# CHECK-LABEL: +print Identifier
print Identifier
# CHECK: "foo"

# CHECK-LABEL: +print OperationName
print OperationName
# CHECK: "FooOp"

# CHECK-LABEL: +print Type
print Type
# CHECK: impl = 0x0

# CHECK-LABEL: +print IndexType
print IndexType
# CHECK: mlir::IndexType

# CHECK-LABEL: +print IntegerType
print IntegerType
# CHECK: mlir::IntegerType
# CHECK: width = 3
# CHECK: Unsigned

# CHECK-LABEL: +print FloatType
print FloatType
# CHECK: mlir::Float32Type

# CHECK-LABEL: +print MemRefType
print MemRefType
# CHECK: mlir::MemRefType
# CHECK: shape = llvm::ArrayRef of length 2 = {4, 5}
# CHECK: elementType
# CHECK: mlir::Float32Type

# CHECK-LABEL: +print UnrankedMemRefType
print UnrankedMemRefType
# CHECK: mlir::UnrankedMemRefType
# CHECK: elementType
# CHECK: mlir::IntegerType
# CHECK: memorySpace
# CHECK: 6

# CHECK-LABEL: +print VectorType
print VectorType
# CHECK: mlir::VectorType
# CHECK: shape = llvm::ArrayRef of length 2 = {1, 2}

# CHECK-LABEL: +print TupleType
print TupleType
# CHECK: mlir::TupleType
# CHECK: numElements = 2
# CHECK: elements[0]
# CHECK: mlir::IndexType
# CHECK: elements[1]
# CHECK: mlir::Float32Type

# CHECK-LABEL: +print Result
print Result
# CHECK: mlir::Float32Type
# CHECK: outOfLineIndex = 42

# CHECK-LABEL: +print Value
print Value
# CHECK: OutOfLineOpResult

# CHECK-LABEL: +print UnknownLoc
print UnknownLoc
# CHECK: mlir::UnknownLoc

# CHECK-LABEL: +print FileLineColLoc
print FileLineColLoc
# CHECK: mlir::FileLineColLoc
# CHECK: "file"
# CHECK: line = 7
# CHECK: column = 8

# CHECK-LABEL: +print OpaqueLoc
print OpaqueLoc
# CHECK: mlir::OpaqueLoc
# CHECK: underlyingLocation = 9

# CHECK-LABEL: +print NameLoc
print NameLoc
# CHECK: mlir::NameLoc
# CHECK: "foo"
# CHECK: mlir::UnknownLoc

# CHECK-LABEL: +print CallSiteLoc
print CallSiteLoc
# CHECK: mlir::CallSiteLoc
# CHECK: callee
# CHECK: mlir::FileLineColLoc
# CHECK: caller
# CHECK: mlir::OpaqueLoc

# CHECK-LABEL: +print FusedLoc
print FusedLoc
# CHECK: mlir::FusedLoc
# CHECK: locations = llvm::ArrayRef of length 2
# CHECK: mlir::FileLineColLoc
# CHECK: mlir::NameLoc

# CHECK-LABEL: +print UnitAttr
print UnitAttr
# CHECK: mlir::UnitAttr

# CHECK-LABEL: +print FloatAttr
print FloatAttr
# CHECK: mlir::FloatAttr

# CHECK-LABEL: +print IntegerAttr
print IntegerAttr
# CHECK: mlir::IntegerAttr

# CHECK-LABEL: +print TypeAttr
print TypeAttr
# CHECK: mlir::TypeAttr
# CHECK: mlir::IndexType

# CHECK-LABEL: +print ArrayAttr
print ArrayAttr
# CHECK: mlir::ArrayAttr
# CHECK: llvm::ArrayRef of length 1
# CHECK: mlir::UnitAttr

# CHECK-LABEL: +print StringAttr
print StringAttr
# CHECK: mlir::StringAttr
# CHECK: value = "foo"

# CHECK-LABEL: +print ElementsAttr
print ElementsAttr
# CHECK: mlir::DenseIntOrFPElementsAttr
