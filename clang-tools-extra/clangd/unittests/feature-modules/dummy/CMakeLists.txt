add_llvm_library(dummyFeatureModule DummyFeatureModule.cpp)
set(FEATURE_MODULES ${FEATURE_MODULES} dummyFeatureModule PARENT_SCOPE)
