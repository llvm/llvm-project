# RUN: gdb -q -batch -n -iex 'source %mlir_src_root/utils/gdb-scripts/prettyprinters.py' -iex 'source %llvm_src_root/utils/gdb-scripts/prettyprinters.py' -x %s %llvm_tools_dir/check-gdb-mlir-support | FileCheck %s
# REQUIRES: debug-info
# REQUIRES: mlir

break main
run

# CHECK: "foo"
p Identifier

# CHECK: "FooOp"
p OperationName

# CHECK: 0x8
# CHECK: TrailingOpResult
p Value

# CHECK: impl = 0x0
p Type

# CHECK: cast<mlir::IndexType>
p IndexType

# CHECK: cast<mlir::IntegerType>
# CHECK: width = 3
# CHECK: Unsigned
p IntegerType

# CHECK: cast<mlir::Float32Type>
p FloatType

# CHECK: cast<mlir::MemRefType>
# CHECK: shapeSize = 2
# CHECK: shapeElements[0] = 4
# CHECK: shapeElements[1] = 5
p MemRefType

# CHECK: cast<mlir::UnrankedMemRefType>
# CHECK: memorySpace = 6
p UnrankedMemRefType

# CHECK: cast<mlir::VectorType>
# CHECK: shapeSize = 2
# CHECK: shapeElements[0] = 1
# CHECK: shapeElements[1] = 2
p VectorType

# CHECK: cast<mlir::TupleType>
# CHECK: numElements = 2
# CHECK: elements[0]
# CHECK: mlir::IndexType
# CHECK: elements[1]
# CHECK: mlir::Float32Type
p TupleType

# CHECK: cast<mlir::UnknownLoc>
p UnknownLoc

# CHECK: cast<mlir::FileLineColLoc>
# CHECK: filename = "file"
# CHECK: line = 7
# CHECK: column = 8
p FileLineColLoc

# CHECK: cast<mlir::OpaqueLoc>
# CHECK: underlyingLocation = 9
p OpaqueLoc

# CHECK: cast<mlir::NameLoc>
# CHECK: name = "foo"
# CHECK: mlir::UnknownLoc
p NameLoc

# CHECK: cast<mlir::CallSiteLoc>
# CHECK: callee
# CHECK: mlir::FileLineColLoc
# CHECK: caller
# CHECK: mlir::OpaqueLoc
p CallSiteLoc

# CHECK: cast<mlir::FusedLoc>
# CHECK: numLocs = 2
# CHECK: locs[0]
# CHECK: mlir::FileLineColLoc
# CHECK: locs[1]
# CHECK: mlir::NameLoc
p FusedLoc

# CHECK: cast<mlir::UnitAttr>
p UnitAttr

# CHECK: cast<mlir::FloatAttr>
p FloatAttr

# CHECK: cast<mlir::IntegerAttr>
p IntegerAttr

# CHECK: cast<mlir::TypeAttr>
# CHECK: mlir::IndexType
p TypeAttr

# CHECK: cast<mlir::ArrayAttr>
# CHECK: llvm::ArrayRef of length 1
# CHECK: mlir::UnitAttr
p ArrayAttr

# CHECK: cast<mlir::StringAttr>
# CHECK: value = "foo"
p StringAttr

# CHECK: cast<mlir::DenseIntOrFPElementsAttr>
p ElementsAttr
