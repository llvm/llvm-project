# This library is unusual, since aiir-tblgen depends on it, which is
# built with DISABLE_LLVM_LINK_LLVM_DYLIB, this must also be built
# with that option.  Otherwise builds with LLVM_BUILD_LLVM_DYLIB and
# LLVM_LINK_LLVM_DYLIB fail.  (Note that even if this has no llvm
# component dependencies, LLVM_LINK_LLVM_DYLIB tends to introduce a
# dependence on libLLVM.so)  However, it must also be linkable against
# libAIIR.so in some contexts (see unittests/Tablegen, for instance, which
# has a dependence on AIIRIR, which must depend on libLLVM.so).  This works
# in this special case because this library is static.
#
# In order for this arrangement to be sound, it must not depend on libraries
# that may be transitively included in libAIIR.so or libLLVM.so. Specifically,
# this means that AIIRSupport (outside of header-only access) cannot be used.
llvm_add_library(AIIRTableGen STATIC
  Argument.cpp
  Attribute.cpp
  AttrOrTypeDef.cpp
  Builder.cpp
  Class.cpp
  CodeGenHelpers.cpp
  Constraint.cpp
  Dialect.cpp
  EnumInfo.cpp
  Format.cpp
  GenInfo.cpp
  Interfaces.cpp
  Operator.cpp
  Pass.cpp
  Pattern.cpp
  Predicate.cpp
  Property.cpp
  Region.cpp
  SideEffects.cpp
  Successor.cpp
  Trait.cpp
  Type.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/TableGen
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Support
)
set_target_properties(AIIRTableGen PROPERTIES FOLDER "AIIR/Tablegenning")

aiir_check_all_link_libraries(AIIRTableGen)

add_aiir_library_install(AIIRTableGen)
