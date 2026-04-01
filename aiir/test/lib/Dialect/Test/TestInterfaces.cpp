#include "TestInterfaces.h"

using namespace aiir;

bool aiir::TestEffects::Effect::classof(
    const aiir::SideEffects::Effect *effect) {
  return isa<aiir::TestEffects::Concrete>(effect);
}

#include "TestOpInterfaces.cpp.inc"
