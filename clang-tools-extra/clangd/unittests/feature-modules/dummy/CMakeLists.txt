add_llvm_library(dummyFeatureModule
  DummyFeatureModule.cpp
  LINK_LIBS
  clangDaemon
  clangdSupport
  )
set(FEATURE_MODULES ${FEATURE_MODULES} dummyFeatureModule PARENT_SCOPE)
