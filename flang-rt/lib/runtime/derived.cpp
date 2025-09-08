//===-- lib/runtime/derived.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/derived.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/stat.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/tools.h"
#include "flang-rt/runtime/type-info.h"
#include "flang-rt/runtime/work-queue.h"

namespace Fortran::runtime {

RT_OFFLOAD_API_GROUP_BEGIN

// Fill "extents" array with the extents of component "comp" from derived type
// instance "derivedInstance".
static RT_API_ATTRS void GetComponentExtents(SubscriptValue (&extents)[maxRank],
    const typeInfo::Component &comp, const Descriptor &derivedInstance) {
  const typeInfo::Value *bounds{comp.bounds()};
  for (int dim{0}; dim < comp.rank(); ++dim) {
    auto lb{bounds[2 * dim].GetValue(&derivedInstance).value_or(0)};
    auto ub{bounds[2 * dim + 1].GetValue(&derivedInstance).value_or(0)};
    extents[dim] = ub >= lb ? static_cast<SubscriptValue>(ub - lb + 1) : 0;
  }
}

RT_API_ATTRS int Initialize(const Descriptor &instance,
    const typeInfo::DerivedType &derived, Terminator &terminator, bool,
    const Descriptor *) {
  WorkQueue workQueue{terminator};
  int status{workQueue.BeginInitialize(instance, derived)};
  return status == StatContinue ? workQueue.Run() : status;
}

RT_API_ATTRS int InitializeTicket::Begin(WorkQueue &) {
  if (elements_ == 0) {
    return StatOk;
  } else {
    // Initialize procedure pointer components in the first element,
    // whence they will be copied later into all others.
    const Descriptor &procPtrDesc{derived_.procPtr()};
    std::size_t numProcPtrs{procPtrDesc.InlineElements()};
    char *raw{instance_.OffsetElement<char>()};
    const auto *ppComponent{
        procPtrDesc.OffsetElement<typeInfo::ProcPtrComponent>()};
    for (std::size_t k{0}; k < numProcPtrs; ++k, ++ppComponent) {
      auto &pptr{*reinterpret_cast<typeInfo::ProcedurePointer *>(
          raw + ppComponent->offset)};
      pptr = ppComponent->procInitialization;
    }
    return StatContinue;
  }
}

RT_API_ATTRS int InitializeTicket::Continue(WorkQueue &workQueue) {
  // Initialize the data components of the first element.
  char *rawInstance{instance_.OffsetElement<char>()};
  for (; !Componentwise::IsComplete(); SkipToNextComponent()) {
    char *rawComponent{rawInstance + component_->offset()};
    if (component_->genre() == typeInfo::Component::Genre::Allocatable) {
      Descriptor &allocDesc{*reinterpret_cast<Descriptor *>(rawComponent)};
      component_->EstablishDescriptor(
          allocDesc, instance_, workQueue.terminator());
    } else if (const void *init{component_->initialization()}) {
      // Explicit initialization of data pointers and
      // non-allocatable non-automatic components
      std::size_t bytes{component_->SizeInBytes(instance_)};
      runtime::memcpy(rawComponent, init, bytes);
    } else if (component_->genre() == typeInfo::Component::Genre::Pointer) {
      // Data pointers without explicit initialization are established
      // so that they are valid right-hand side targets of pointer
      // assignment statements.
      Descriptor &ptrDesc{*reinterpret_cast<Descriptor *>(rawComponent)};
      component_->EstablishDescriptor(
          ptrDesc, instance_, workQueue.terminator());
    } else if (component_->genre() == typeInfo::Component::Genre::Data &&
        component_->derivedType() &&
        !component_->derivedType()->noInitializationNeeded()) {
      // Default initialization of non-pointer non-allocatable/automatic
      // data component.  Handles parent component's elements.
      SubscriptValue extents[maxRank];
      GetComponentExtents(extents, *component_, instance_);
      Descriptor &compDesc{componentDescriptor_.descriptor()};
      const typeInfo::DerivedType &compType{*component_->derivedType()};
      compDesc.Establish(compType, rawComponent, component_->rank(), extents);
      if (int status{workQueue.BeginInitialize(compDesc, compType)};
          status != StatOk) {
        SkipToNextComponent();
        return status;
      }
    }
  }
  // The first element is now complete.  Copy it into the others.
  if (elements_ < 2) {
  } else {
    auto elementBytes{static_cast<SubscriptValue>(instance_.ElementBytes())};
    if (auto stride{instance_.FixedStride()}) {
      if (*stride == elementBytes) { // contiguous
        for (std::size_t done{1}; done < elements_;) {
          std::size_t chunk{elements_ - done};
          if (chunk > done) {
            chunk = done;
          }
          char *uninitialized{rawInstance + done * *stride};
          runtime::memcpy(uninitialized, rawInstance, chunk * *stride);
          done += chunk;
        }
      } else {
        for (std::size_t done{1}; done < elements_; ++done) {
          char *uninitialized{rawInstance + done * *stride};
          runtime::memcpy(uninitialized, rawInstance, elementBytes);
        }
      }
    } else { // one at a time with subscription
      for (Elementwise::Advance(); !Elementwise::IsComplete();
          Elementwise::Advance()) {
        char *element{instance_.Element<char>(subscripts_)};
        runtime::memcpy(element, rawInstance, elementBytes);
      }
    }
  }
  return StatOk;
}

RT_API_ATTRS int InitializeClone(const Descriptor &clone,
    const Descriptor &original, const typeInfo::DerivedType &derived,
    Terminator &terminator, bool hasStat, const Descriptor *errMsg) {
  if (original.IsPointer() || !original.IsAllocated()) {
    return StatOk; // nothing to do
  } else {
    WorkQueue workQueue{terminator};
    int status{workQueue.BeginInitializeClone(
        clone, original, derived, hasStat, errMsg)};
    return status == StatContinue ? workQueue.Run() : status;
  }
}

RT_API_ATTRS int InitializeCloneTicket::Continue(WorkQueue &workQueue) {
  while (!IsComplete()) {
    if (component_->genre() == typeInfo::Component::Genre::Allocatable) {
      Descriptor &origDesc{*instance_.ElementComponent<Descriptor>(
          subscripts_, component_->offset())};
      if (origDesc.IsAllocated()) {
        Descriptor &cloneDesc{*clone_.ElementComponent<Descriptor>(
            subscripts_, component_->offset())};
        if (phase_ == 0) {
          ++phase_;
          cloneDesc.ApplyMold(origDesc, origDesc.rank());
          if (int stat{ReturnError(workQueue.terminator(),
                  cloneDesc.Allocate(kNoAsyncObject), errMsg_, hasStat_)};
              stat != StatOk) {
            return stat;
          }
          if (const DescriptorAddendum *addendum{cloneDesc.Addendum()}) {
            if (const typeInfo::DerivedType *derived{addendum->derivedType()}) {
              if (!derived->noInitializationNeeded()) {
                // Perform default initialization for the allocated element.
                if (int status{workQueue.BeginInitialize(cloneDesc, *derived)};
                    status != StatOk) {
                  return status;
                }
              }
            }
          }
        }
        if (phase_ == 1) {
          ++phase_;
          if (const DescriptorAddendum *addendum{cloneDesc.Addendum()}) {
            if (const typeInfo::DerivedType *derived{addendum->derivedType()}) {
              // Initialize derived type's allocatables.
              if (int status{workQueue.BeginInitializeClone(
                      cloneDesc, origDesc, *derived, hasStat_, errMsg_)};
                  status != StatOk) {
                return status;
              }
            }
          }
        }
      }
      Advance();
    } else if (component_->genre() == typeInfo::Component::Genre::Data) {
      if (component_->derivedType()) {
        // Handle nested derived types.
        const typeInfo::DerivedType &compType{*component_->derivedType()};
        SubscriptValue extents[maxRank];
        GetComponentExtents(extents, *component_, instance_);
        Descriptor &origDesc{componentDescriptor_.descriptor()};
        Descriptor &cloneDesc{cloneComponentDescriptor_.descriptor()};
        origDesc.Establish(compType,
            instance_.ElementComponent<char>(subscripts_, component_->offset()),
            component_->rank(), extents);
        cloneDesc.Establish(compType,
            clone_.ElementComponent<char>(subscripts_, component_->offset()),
            component_->rank(), extents);
        Advance();
        if (int status{workQueue.BeginInitializeClone(
                cloneDesc, origDesc, compType, hasStat_, errMsg_)};
            status != StatOk) {
          return status;
        }
      } else {
        SkipToNextComponent();
      }
    } else {
      SkipToNextComponent();
    }
  }
  return StatOk;
}

// Fortran 2018 subclause 7.5.6.2
RT_API_ATTRS void Finalize(const Descriptor &descriptor,
    const typeInfo::DerivedType &derived, Terminator *terminator) {
  if (!derived.noFinalizationNeeded() && descriptor.IsAllocated()) {
    Terminator stubTerminator{"Finalize() in Fortran runtime", 0};
    WorkQueue workQueue{terminator ? *terminator : stubTerminator};
    if (workQueue.BeginFinalize(descriptor, derived) == StatContinue) {
      workQueue.Run();
    }
  }
}

static RT_API_ATTRS const typeInfo::SpecialBinding *FindFinal(
    const typeInfo::DerivedType &derived, int rank) {
  if (const auto *ranked{derived.FindSpecialBinding(
          typeInfo::SpecialBinding::RankFinal(rank))}) {
    return ranked;
  } else if (const auto *assumed{derived.FindSpecialBinding(
                 typeInfo::SpecialBinding::Which::AssumedRankFinal)}) {
    return assumed;
  } else {
    return derived.FindSpecialBinding(
        typeInfo::SpecialBinding::Which::ElementalFinal);
  }
}

static RT_API_ATTRS void CallFinalSubroutine(const Descriptor &descriptor,
    const typeInfo::DerivedType &derived, Terminator &terminator) {
  if (const auto *special{FindFinal(derived, descriptor.rank())}) {
    if (special->which() == typeInfo::SpecialBinding::Which::ElementalFinal) {
      std::size_t elements{descriptor.InlineElements()};
      SubscriptValue at[maxRank];
      descriptor.GetLowerBounds(at);
      if (special->IsArgDescriptor(0)) {
        StaticDescriptor<maxRank, true, 8 /*?*/> statDesc;
        Descriptor &elemDesc{statDesc.descriptor()};
        elemDesc = descriptor;
        elemDesc.raw().attribute = CFI_attribute_pointer;
        elemDesc.raw().rank = 0;
        auto *p{special->GetProc<void (*)(const Descriptor &)>()};
        for (std::size_t j{0}; j++ < elements;
             descriptor.IncrementSubscripts(at)) {
          elemDesc.set_base_addr(descriptor.Element<char>(at));
          p(elemDesc);
        }
      } else {
        auto *p{special->GetProc<void (*)(char *)>()};
        for (std::size_t j{0}; j++ < elements;
             descriptor.IncrementSubscripts(at)) {
          p(descriptor.Element<char>(at));
        }
      }
    } else {
      StaticDescriptor<maxRank, true, 10> statDesc;
      Descriptor &copy{statDesc.descriptor()};
      const Descriptor *argDescriptor{&descriptor};
      if (descriptor.rank() > 0 && special->specialCaseFlag() &&
          !descriptor.IsContiguous()) {
        // The FINAL subroutine demands a contiguous array argument, but
        // this INTENT(OUT) or intrinsic assignment LHS isn't contiguous.
        // Finalize a shallow copy of the data.
        copy = descriptor;
        copy.set_base_addr(nullptr);
        copy.raw().attribute = CFI_attribute_allocatable;
        RUNTIME_CHECK(terminator, copy.Allocate(kNoAsyncObject) == CFI_SUCCESS);
        ShallowCopyDiscontiguousToContiguous(copy, descriptor);
        argDescriptor = &copy;
      }
      if (special->IsArgDescriptor(0)) {
        StaticDescriptor<maxRank, true, 8 /*?*/> statDesc;
        Descriptor &tmpDesc{statDesc.descriptor()};
        tmpDesc = *argDescriptor;
        tmpDesc.raw().attribute = CFI_attribute_pointer;
        tmpDesc.Addendum()->set_derivedType(&derived);
        auto *p{special->GetProc<void (*)(const Descriptor &)>()};
        p(tmpDesc);
      } else {
        auto *p{special->GetProc<void (*)(char *)>()};
        p(argDescriptor->OffsetElement<char>());
      }
      if (argDescriptor == &copy) {
        ShallowCopyContiguousToDiscontiguous(descriptor, copy);
        copy.Deallocate();
      }
    }
  }
}

RT_API_ATTRS int FinalizeTicket::Begin(WorkQueue &workQueue) {
  CallFinalSubroutine(instance_, derived_, workQueue.terminator());
  // If there's a finalizable parent component, handle it last, as required
  // by the Fortran standard (7.5.6.2), and do so recursively with the same
  // descriptor so that the rank is preserved.
  finalizableParentType_ = derived_.GetParentType();
  if (finalizableParentType_) {
    if (finalizableParentType_->noFinalizationNeeded()) {
      finalizableParentType_ = nullptr;
    } else {
      SkipToNextComponent();
    }
  }
  return StatContinue;
}

RT_API_ATTRS int FinalizeTicket::Continue(WorkQueue &workQueue) {
  while (!IsComplete()) {
    if (component_->genre() == typeInfo::Component::Genre::Allocatable &&
        component_->category() == TypeCategory::Derived) {
      // Component may be polymorphic or unlimited polymorphic. Need to use the
      // dynamic type to check whether finalization is needed.
      const Descriptor &compDesc{*instance_.ElementComponent<Descriptor>(
          subscripts_, component_->offset())};
      Advance();
      if (compDesc.IsAllocated()) {
        if (const DescriptorAddendum *addendum{compDesc.Addendum()}) {
          if (const typeInfo::DerivedType *compDynamicType{
                  addendum->derivedType()}) {
            if (!compDynamicType->noFinalizationNeeded()) {
              if (int status{
                      workQueue.BeginFinalize(compDesc, *compDynamicType)};
                  status != StatOk) {
                return status;
              }
            }
          }
        }
      }
    } else if (component_->genre() == typeInfo::Component::Genre::Allocatable ||
        component_->genre() == typeInfo::Component::Genre::Automatic) {
      if (const typeInfo::DerivedType *compType{component_->derivedType()};
          compType && !compType->noFinalizationNeeded()) {
        const Descriptor &compDesc{*instance_.ElementComponent<Descriptor>(
            subscripts_, component_->offset())};
        Advance();
        if (compDesc.IsAllocated()) {
          if (int status{workQueue.BeginFinalize(compDesc, *compType)};
              status != StatOk) {
            return status;
          }
        }
      } else {
        SkipToNextComponent();
      }
    } else if (component_->genre() == typeInfo::Component::Genre::Data &&
        component_->derivedType() &&
        !component_->derivedType()->noFinalizationNeeded()) {
      // todo: calculate and use fixedStride_ here as in DestroyTicket to
      // avoid subscripts and repeated descriptor establishment.
      SubscriptValue extents[maxRank];
      GetComponentExtents(extents, *component_, instance_);
      Descriptor &compDesc{componentDescriptor_.descriptor()};
      const typeInfo::DerivedType &compType{*component_->derivedType()};
      compDesc.Establish(compType,
          instance_.ElementComponent<char>(subscripts_, component_->offset()),
          component_->rank(), extents);
      Advance();
      if (int status{workQueue.BeginFinalize(compDesc, compType)};
          status != StatOk) {
        return status;
      }
    } else {
      SkipToNextComponent();
    }
  }
  // Last, do the parent component, if any and finalizable.
  if (finalizableParentType_) {
    Descriptor &tmpDesc{componentDescriptor_.descriptor()};
    tmpDesc = instance_;
    tmpDesc.raw().attribute = CFI_attribute_pointer;
    tmpDesc.Addendum()->set_derivedType(finalizableParentType_);
    tmpDesc.raw().elem_len = finalizableParentType_->sizeInBytes();
    const auto &parentType{*finalizableParentType_};
    finalizableParentType_ = nullptr;
    // Don't return StatOk here if the nested FInalize is still running;
    // it needs this->componentDescriptor_.
    return workQueue.BeginFinalize(tmpDesc, parentType);
  }
  return StatOk;
}

// The order of finalization follows Fortran 2018 7.5.6.2, with
// elementwise finalization of non-parent components taking place
// before parent component finalization, and with all finalization
// preceding any deallocation.
RT_API_ATTRS void Destroy(const Descriptor &descriptor, bool finalize,
    const typeInfo::DerivedType &derived, Terminator *terminator) {
  if (descriptor.IsAllocated() && !derived.noDestructionNeeded()) {
    Terminator stubTerminator{"Destroy() in Fortran runtime", 0};
    WorkQueue workQueue{terminator ? *terminator : stubTerminator};
    if (workQueue.BeginDestroy(descriptor, derived, finalize) == StatContinue) {
      workQueue.Run();
    }
  }
}

RT_API_ATTRS int DestroyTicket::Begin(WorkQueue &workQueue) {
  if (finalize_ && !derived_.noFinalizationNeeded()) {
    if (int status{workQueue.BeginFinalize(instance_, derived_)};
        status != StatOk && status != StatContinue) {
      return status;
    }
  }
  return StatContinue;
}

RT_API_ATTRS int DestroyTicket::Continue(WorkQueue &workQueue) {
  // Deallocate all direct and indirect allocatable and automatic components.
  // Contrary to finalization, the order of deallocation does not matter.
  while (!IsComplete()) {
    const auto *componentDerived{component_->derivedType()};
    if (component_->genre() == typeInfo::Component::Genre::Allocatable) {
      if (fixedStride_ &&
          (!componentDerived || componentDerived->noDestructionNeeded())) {
        // common fast path, just deallocate in every element
        char *p{instance_.OffsetElement<char>(component_->offset())};
        for (std::size_t j{0}; j < elements_; ++j, p += *fixedStride_) {
          Descriptor &d{*reinterpret_cast<Descriptor *>(p)};
          d.Deallocate();
        }
        SkipToNextComponent();
      } else {
        Descriptor &d{*instance_.ElementComponent<Descriptor>(
            subscripts_, component_->offset())};
        if (d.IsAllocated()) {
          if (componentDerived && !componentDerived->noDestructionNeeded() &&
              phase_ == 0) {
            if (int status{workQueue.BeginDestroy(
                    d, *componentDerived, /*finalize=*/false)};
                status != StatOk) {
              ++phase_;
              return status;
            }
          }
          d.Deallocate();
        }
        Advance();
      }
    } else if (component_->genre() == typeInfo::Component::Genre::Data) {
      if (!componentDerived || componentDerived->noDestructionNeeded()) {
        SkipToNextComponent();
      } else if (fixedStride_) {
        // faster path, no need for subscripts, can reuse descriptor
        char *p{instance_.OffsetElement<char>(
            elementAt_ * *fixedStride_ + component_->offset())};
        Descriptor &compDesc{componentDescriptor_.descriptor()};
        const typeInfo::DerivedType &compType{*componentDerived};
        compDesc.UncheckedScalarEstablish(compType, p);
        for (std::size_t j{elementAt_}; j < elements_;
            ++j, p += *fixedStride_) {
          compDesc.set_base_addr(p);
          ++elementAt_;
          if (int status{workQueue.BeginDestroy(
                  compDesc, compType, /*finalize=*/false)};
              status != StatOk) {
            return status;
          }
        }
        SkipToNextComponent();
      } else {
        SubscriptValue extents[maxRank];
        GetComponentExtents(extents, *component_, instance_);
        Descriptor &compDesc{componentDescriptor_.descriptor()};
        const typeInfo::DerivedType &compType{*componentDerived};
        compDesc.Establish(compType,
            instance_.ElementComponent<char>(subscripts_, component_->offset()),
            component_->rank(), extents);
        Advance();
        if (int status{
                workQueue.BeginDestroy(compDesc, compType, /*finalize=*/false)};
            status != StatOk) {
          return status;
        }
      }
    } else {
      SkipToNextComponent();
    }
  }
  return StatOk;
}

RT_API_ATTRS bool HasDynamicComponent(const Descriptor &descriptor) {
  if (const DescriptorAddendum * addendum{descriptor.Addendum()}) {
    if (const auto *derived = addendum->derivedType()) {
      // Destruction is needed if and only if there are direct or indirect
      // allocatable or automatic components.
      return !derived->noDestructionNeeded();
    }
  }
  return false;
}

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime
