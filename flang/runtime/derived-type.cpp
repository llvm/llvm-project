//===-- runtime/derived-type.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "derived-type.h"
#include "descriptor.h"
#include <cstring>

namespace Fortran::runtime {

TypeParameterValue TypeParameter::GetValue(const Descriptor &descriptor) const {
  if (which_ < 0) {
    return value_;
  } else {
    return descriptor.Addendum()->LenParameterValue(which_);
  }
}

bool DerivedType::IsNontrivialAnalysis() const {
  if (kindParameters_ > 0 || lenParameters_ > 0 || typeBoundProcedures_ > 0) {
    return true;
  }
  for (std::size_t j{0}; j < components_; ++j) {
    if (component_[j].IsDescriptor()) {
      return true;
    }
    if (const Descriptor * staticDescriptor{component_[j].staticDescriptor()}) {
      if (const DescriptorAddendum * addendum{staticDescriptor->Addendum()}) {
        if (const DerivedType * dt{addendum->derivedType()}) {
          if (dt->IsNontrivial()) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

void DerivedType::Initialize(char *instance) const {
  if (typeBoundProcedures_ > InitializerTBP) {
    if (auto f{reinterpret_cast<void (*)(char *)>(
            typeBoundProcedure_[InitializerTBP].code.host)}) {
      f(instance);
    }
  }
#if 0 // TODO
  for (std::size_t j{0}; j < components_; ++j) {
    if (const Descriptor * descriptor{component_[j].GetDescriptor(instance)}) {
      // invoke initialization TBP
    }
  }
#endif
}

void DerivedType::Destroy(char *instance, bool finalize) const {
  if (finalize && typeBoundProcedures_ > FinalTBP) {
    if (auto f{reinterpret_cast<void (*)(char *)>(
            typeBoundProcedure_[FinalTBP].code.host)}) {
      f(instance);
    }
  }
  const char *constInstance{instance};
  for (std::size_t j{0}; j < components_; ++j) {
    if (Descriptor * descriptor{component_[j].GetDescriptor(instance)}) {
      descriptor->Deallocate(finalize);
    } else if (const Descriptor *
        descriptor{component_[j].GetDescriptor(constInstance)}) {
      descriptor->Destroy(component_[j].Locate<char>(instance), finalize);
    }
  }
}
} // namespace Fortran::runtime
