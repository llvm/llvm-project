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
# CHECK: typeID = mlir::TypeID::get<mlir::IndexType>()

# CHECK-LABEL: +print IntegerType
print IntegerType
# CHECK: typeID = mlir::TypeID::get<mlir::IntegerType>()
# CHECK: members of mlir::detail::IntegerTypeStorage

# CHECK-LABEL: +print FloatType
print FloatType
# CHECK: typeID = mlir::TypeID::get<mlir::Float32Type>()

# CHECK-LABEL: +print MemRefType
print MemRefType
# CHECK: typeID = mlir::TypeID::get<mlir::MemRefType>()
# CHECK: members of mlir::detail::MemRefTypeStorage

# CHECK-LABEL: +print UnrankedMemRefType
print UnrankedMemRefType
# CHECK: typeID = mlir::TypeID::get<mlir::UnrankedMemRefType>()
# CHECK: members of mlir::detail::UnrankedMemRefTypeStorage

# CHECK-LABEL: +print VectorType
print VectorType
# CHECK: typeID = mlir::TypeID::get<mlir::VectorType>()
# CHECK: members of mlir::detail::VectorTypeStorage

# CHECK-LABEL: +print TupleType
print TupleType
# CHECK: typeID = mlir::TypeID::get<mlir::TupleType>()
# CHECK: elements[0]
# CHECK-NEXT: typeID = mlir::TypeID::get<mlir::IndexType>()
# CHECK: elements[1]
# CHECK-NEXT: typeID = mlir::TypeID::get<mlir::Float32Type>()

# CHECK-LABEL: +print Result
print Result
# CHECK: typeID = mlir::TypeID::get<mlir::Float32Type>()
# CHECK: outOfLineIndex = 42

# CHECK-LABEL: +print Value
print Value
# CHECK: typeID = mlir::TypeID::get<mlir::Float32Type>()
# CHECK: mlir::detail::ValueImpl::Kind::OutOfLineOpResult

# CHECK-LABEL: +print UnknownLoc
print UnknownLoc
# CHECK: typeID = mlir::TypeID::get<mlir::UnknownLoc>()

# CHECK-LABEL: +print FileLineColLoc
print FileLineColLoc
# CHECK: typeID = mlir::TypeID::get<mlir::FileLineColLoc>()
# CHECK: members of mlir::detail::FileLineColLocAttrStorage
# CHECK: "file"
# CHECK: line = 7
# CHECK: column = 8

# CHECK-LABEL: +print OpaqueLoc
print OpaqueLoc
# CHECK: typeID = mlir::TypeID::get<mlir::OpaqueLoc>()
# CHECK: members of mlir::detail::OpaqueLocAttrStorage
# CHECK: underlyingLocation = 9

# CHECK-LABEL: +print NameLoc
print NameLoc
# CHECK: typeID = mlir::TypeID::get<mlir::NameLoc>()
# CHECK: members of mlir::detail::NameLocAttrStorage
# CHECK: "foo"
# CHECK: typeID = mlir::TypeID::get<mlir::UnknownLoc>()

# CHECK-LABEL: +print CallSiteLoc
print CallSiteLoc
# CHECK: typeID = mlir::TypeID::get<mlir::CallSiteLoc>()
# CHECK: members of mlir::detail::CallSiteLocAttrStorage
# CHECK: typeID = mlir::TypeID::get<mlir::FileLineColLoc>()
# CHECK: typeID = mlir::TypeID::get<mlir::OpaqueLoc>()

# CHECK-LABEL: +print FusedLoc
print FusedLoc
# CHECK: typeID = mlir::TypeID::get<mlir::FusedLoc>()
# CHECK: members of mlir::detail::FusedLocAttrStorage
# CHECK: locations = llvm::ArrayRef of length 2
# CHECK: typeID = mlir::TypeID::get<mlir::FileLineColLoc>()
# CHECK: typeID = mlir::TypeID::get<mlir::NameLoc>()

# CHECK-LABEL: +print UnitAttr
print UnitAttr
# CHECK: typeID = mlir::TypeID::get<mlir::UnitAttr>()

# CHECK-LABEL: +print FloatAttr
print FloatAttr
# CHECK: typeID = mlir::TypeID::get<mlir::FloatAttr>()
# CHECK: members of mlir::detail::FloatAttrStorage

# CHECK-LABEL: +print IntegerAttr
print IntegerAttr
# CHECK: typeID = mlir::TypeID::get<mlir::IntegerAttr>()
# CHECK: members of mlir::detail::IntegerAttrStorage

# CHECK-LABEL: +print TypeAttr
print TypeAttr
# CHECK: typeID = mlir::TypeID::get<mlir::TypeAttr>()
# CHECK: members of mlir::detail::TypeAttrStorage
# CHECK: typeID = mlir::TypeID::get<mlir::IndexType>()

# CHECK-LABEL: +print ArrayAttr
print ArrayAttr
# CHECK: typeID = mlir::TypeID::get<mlir::ArrayAttr>()
# CHECK: members of mlir::detail::ArrayAttrStorage
# CHECK: llvm::ArrayRef of length 1
# CHECK: typeID = mlir::TypeID::get<mlir::UnitAttr>()

# CHECK-LABEL: +print StringAttr
print StringAttr
# CHECK: typeID = mlir::TypeID::get<mlir::StringAttr>()
# CHECK: members of mlir::detail::StringAttrStorage
# CHECK: value = "foo"

# CHECK-LABEL: +print ElementsAttr
print ElementsAttr
# CHECK: typeID = mlir::TypeID::get<mlir::DenseIntOrFPElementsAttr>()
# CHECK: members of mlir::detail::DenseIntOrFPElementsAttrStorage
