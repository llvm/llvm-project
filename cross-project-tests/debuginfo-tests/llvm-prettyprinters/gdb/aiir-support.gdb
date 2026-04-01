# RUN: gdb -q -batch -n \
# RUN:   -iex 'source %aiir_src_root/utils/gdb-scripts/prettyprinters.py' \
# RUN:   -iex 'source %llvm_src_root/utils/gdb-scripts/prettyprinters.py' \
# RUN:   -ex 'source -v %s' %llvm_tools_dir/check-gdb-aiir-support \
# RUN: | FileCheck %s
# REQUIRES: debug-info
# REQUIRES: aiir

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
# CHECK: typeID = aiir::TypeID::get<aiir::IndexType>()

# CHECK-LABEL: +print IntegerType
print IntegerType
# CHECK: typeID = aiir::TypeID::get<aiir::IntegerType>()
# CHECK: members of aiir::detail::IntegerTypeStorage

# CHECK-LABEL: +print FloatType
print FloatType
# CHECK: typeID = aiir::TypeID::get<aiir::Float32Type>()

# CHECK-LABEL: +print MemRefType
print MemRefType
# CHECK: typeID = aiir::TypeID::get<aiir::MemRefType>()
# CHECK: members of aiir::detail::MemRefTypeStorage

# CHECK-LABEL: +print UnrankedMemRefType
print UnrankedMemRefType
# CHECK: typeID = aiir::TypeID::get<aiir::UnrankedMemRefType>()
# CHECK: members of aiir::detail::UnrankedMemRefTypeStorage

# CHECK-LABEL: +print VectorType
print VectorType
# CHECK: typeID = aiir::TypeID::get<aiir::VectorType>()
# CHECK: members of aiir::detail::VectorTypeStorage

# CHECK-LABEL: +print TupleType
print TupleType
# CHECK: typeID = aiir::TypeID::get<aiir::TupleType>()
# CHECK: elements[0]
# CHECK-NEXT: typeID = aiir::TypeID::get<aiir::IndexType>()
# CHECK: elements[1]
# CHECK-NEXT: typeID = aiir::TypeID::get<aiir::Float32Type>()

# CHECK-LABEL: +print Result
print Result
# CHECK: typeID = aiir::TypeID::get<aiir::Float32Type>()
# CHECK: outOfLineIndex = 42

# CHECK-LABEL: +print Value
print Value
# CHECK: typeID = aiir::TypeID::get<aiir::Float32Type>()
# CHECK: aiir::detail::ValueImpl::Kind::OutOfLineOpResult

# CHECK-LABEL: +print UnknownLoc
print UnknownLoc
# CHECK: typeID = aiir::TypeID::get<aiir::UnknownLoc>()

# CHECK-LABEL: +print FileLineColLoc
print FileLineColLoc
# CHECK: typeID = aiir::TypeID::get<aiir::FileLineColLoc>()
# CHECK: members of aiir::detail::FileLineColLocAttrStorage
# CHECK: "file"
# CHECK: line = 7
# CHECK: column = 8

# CHECK-LABEL: +print OpaqueLoc
print OpaqueLoc
# CHECK: typeID = aiir::TypeID::get<aiir::OpaqueLoc>()
# CHECK: members of aiir::detail::OpaqueLocAttrStorage
# CHECK: underlyingLocation = 9

# CHECK-LABEL: +print NameLoc
print NameLoc
# CHECK: typeID = aiir::TypeID::get<aiir::NameLoc>()
# CHECK: members of aiir::detail::NameLocAttrStorage
# CHECK: "foo"
# CHECK: typeID = aiir::TypeID::get<aiir::UnknownLoc>()

# CHECK-LABEL: +print CallSiteLoc
print CallSiteLoc
# CHECK: typeID = aiir::TypeID::get<aiir::CallSiteLoc>()
# CHECK: members of aiir::detail::CallSiteLocAttrStorage
# CHECK: typeID = aiir::TypeID::get<aiir::FileLineColLoc>()
# CHECK: typeID = aiir::TypeID::get<aiir::OpaqueLoc>()

# CHECK-LABEL: +print FusedLoc
print FusedLoc
# CHECK: typeID = aiir::TypeID::get<aiir::FusedLoc>()
# CHECK: members of aiir::detail::FusedLocAttrStorage
# CHECK: locations = llvm::ArrayRef of length 2
# CHECK: typeID = aiir::TypeID::get<aiir::FileLineColLoc>()
# CHECK: typeID = aiir::TypeID::get<aiir::NameLoc>()

# CHECK-LABEL: +print UnitAttr
print UnitAttr
# CHECK: typeID = aiir::TypeID::get<aiir::UnitAttr>()

# CHECK-LABEL: +print FloatAttr
print FloatAttr
# CHECK: typeID = aiir::TypeID::get<aiir::FloatAttr>()
# CHECK: members of aiir::detail::FloatAttrStorage

# CHECK-LABEL: +print IntegerAttr
print IntegerAttr
# CHECK: typeID = aiir::TypeID::get<aiir::IntegerAttr>()
# CHECK: members of aiir::detail::IntegerAttrStorage

# CHECK-LABEL: +print TypeAttr
print TypeAttr
# CHECK: typeID = aiir::TypeID::get<aiir::TypeAttr>()
# CHECK: members of aiir::detail::TypeAttrStorage
# CHECK: typeID = aiir::TypeID::get<aiir::IndexType>()

# CHECK-LABEL: +print ArrayAttr
print ArrayAttr
# CHECK: typeID = aiir::TypeID::get<aiir::ArrayAttr>()
# CHECK: members of aiir::detail::ArrayAttrStorage
# CHECK: llvm::ArrayRef of length 1
# CHECK: typeID = aiir::TypeID::get<aiir::UnitAttr>()

# CHECK-LABEL: +print StringAttr
print StringAttr
# CHECK: typeID = aiir::TypeID::get<aiir::StringAttr>()
# CHECK: members of aiir::detail::StringAttrStorage
# CHECK: value = "foo"

# CHECK-LABEL: +print ElementsAttr
print ElementsAttr
# CHECK: typeID = aiir::TypeID::get<aiir::DenseTypedElementsAttr>()
# CHECK: members of aiir::detail::DenseTypedElementsAttrStorage
