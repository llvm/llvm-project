//===-- runtime/type-info.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "type-info.h"
#include "terminator.h"
#include <cstdio>

namespace Fortran::runtime::typeInfo {

std::optional<TypeParameterValue> Value::GetValue(
    const Descriptor *descriptor) const {
  switch (genre_) {
  case Genre::Explicit:
    return value_;
  case Genre::LenParameter:
    if (descriptor) {
      if (const auto *addendum{descriptor->Addendum()}) {
        return addendum->LenParameterValue(value_);
      }
    }
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

std::size_t Component::GetElementByteSize(const Descriptor &instance) const {
  switch (category()) {
  case TypeCategory::Integer:
  case TypeCategory::Real:
  case TypeCategory::Logical:
    return kind_;
  case TypeCategory::Complex:
    return 2 * kind_;
  case TypeCategory::Character:
    if (auto value{characterLen_.GetValue(&instance)}) {
      return kind_ * *value;
    }
    break;
  case TypeCategory::Derived:
    if (const auto *type{derivedType()}) {
      return type->sizeInBytes();
    }
    break;
  }
  return 0;
}

std::size_t Component::GetElements(const Descriptor &instance) const {
  std::size_t elements{1};
  if (int rank{rank_}) {
    if (const Value * boundValues{bounds()}) {
      for (int j{0}; j < rank; ++j) {
        TypeParameterValue lb{
            boundValues[2 * j].GetValue(&instance).value_or(0)};
        TypeParameterValue ub{
            boundValues[2 * j + 1].GetValue(&instance).value_or(0)};
        if (ub >= lb) {
          elements *= ub - lb + 1;
        } else {
          return 0;
        }
      }
    } else {
      return 0;
    }
  }
  return elements;
}

std::size_t Component::SizeInBytes(const Descriptor &instance) const {
  if (genre() == Genre::Data) {
    return GetElementByteSize(instance) * GetElements(instance);
  } else if (category() == TypeCategory::Derived) {
    const DerivedType *type{derivedType()};
    return Descriptor::SizeInBytes(
        rank_, true, type ? type->LenParameters() : 0);
  } else {
    return Descriptor::SizeInBytes(rank_);
  }
}

void Component::EstablishDescriptor(Descriptor &descriptor,
    const Descriptor &container, Terminator &terminator) const {
  TypeCategory cat{category()};
  if (cat == TypeCategory::Character) {
    auto length{characterLen_.GetValue(&container)};
    RUNTIME_CHECK(terminator, length.has_value());
    descriptor.Establish(kind_, *length / kind_, nullptr, rank_);
  } else if (cat == TypeCategory::Derived) {
    const DerivedType *type{derivedType()};
    RUNTIME_CHECK(terminator, type != nullptr);
    descriptor.Establish(*type, nullptr, rank_);
  } else {
    descriptor.Establish(cat, kind_, nullptr, rank_);
  }
  if (rank_ && genre_ != Genre::Allocatable) {
    const typeInfo::Value *boundValues{bounds()};
    RUNTIME_CHECK(terminator, boundValues != nullptr);
    auto byteStride{static_cast<SubscriptValue>(descriptor.ElementBytes())};
    for (int j{0}; j < rank_; ++j) {
      auto lb{boundValues++->GetValue(&container)};
      auto ub{boundValues++->GetValue(&container)};
      RUNTIME_CHECK(terminator, lb.has_value() && ub.has_value());
      Dimension &dim{descriptor.GetDimension(j)};
      dim.SetBounds(*lb, *ub);
      dim.SetByteStride(byteStride);
      byteStride *= dim.Extent();
    }
  }
}

void Component::CreatePointerDescriptor(Descriptor &descriptor,
    const Descriptor &container, Terminator &terminator,
    const SubscriptValue *subscripts) const {
  RUNTIME_CHECK(terminator, genre_ == Genre::Data);
  EstablishDescriptor(descriptor, container, terminator);
  if (subscripts) {
    descriptor.set_base_addr(container.Element<char>(subscripts) + offset_);
  } else {
    descriptor.set_base_addr(container.OffsetElement<char>() + offset_);
  }
  descriptor.raw().attribute = CFI_attribute_pointer;
}

const DerivedType *DerivedType::GetParentType() const {
  if (hasParent_) {
    const Descriptor &compDesc{component()};
    const Component &component{*compDesc.OffsetElement<const Component>()};
    return component.derivedType();
  } else {
    return nullptr;
  }
}

const Component *DerivedType::FindDataComponent(
    const char *compName, std::size_t compNameLen) const {
  const Descriptor &compDesc{component()};
  std::size_t n{compDesc.Elements()};
  SubscriptValue at[maxRank];
  compDesc.GetLowerBounds(at);
  for (std::size_t j{0}; j < n; ++j, compDesc.IncrementSubscripts(at)) {
    const Component *component{compDesc.Element<Component>(at)};
    INTERNAL_CHECK(component != nullptr);
    const Descriptor &nameDesc{component->name()};
    if (nameDesc.ElementBytes() == compNameLen &&
        std::memcmp(compName, nameDesc.OffsetElement(), compNameLen) == 0) {
      return component;
    }
  }
  const DerivedType *parent{GetParentType()};
  return parent ? parent->FindDataComponent(compName, compNameLen) : nullptr;
}

static void DumpScalarCharacter(
    FILE *f, const Descriptor &desc, const char *what) {
  if (desc.raw().version == CFI_VERSION &&
      desc.type() == TypeCode{TypeCategory::Character, 1} &&
      desc.ElementBytes() > 0 && desc.rank() == 0 &&
      desc.OffsetElement() != nullptr) {
    std::fwrite(desc.OffsetElement(), desc.ElementBytes(), 1, f);
  } else {
    std::fprintf(f, "bad %s descriptor: ", what);
    desc.Dump(f);
  }
}

FILE *DerivedType::Dump(FILE *f) const {
  std::fprintf(f, "DerivedType @ %p:\n", reinterpret_cast<const void *>(this));
  const std::uint64_t *uints{reinterpret_cast<const std::uint64_t *>(this)};
  for (int j{0}; j < 64; ++j) {
    int offset{j * static_cast<int>(sizeof *uints)};
    std::fprintf(f, "    [+%3d](%p) 0x%016jx", offset,
        reinterpret_cast<const void *>(&uints[j]),
        static_cast<std::uintmax_t>(uints[j]));
    if (offset == offsetof(DerivedType, binding_)) {
      std::fputs(" <-- binding_\n", f);
    } else if (offset == offsetof(DerivedType, name_)) {
      std::fputs(" <-- name_\n", f);
    } else if (offset == offsetof(DerivedType, sizeInBytes_)) {
      std::fputs(" <-- sizeInBytes_\n", f);
    } else if (offset == offsetof(DerivedType, uninstantiated_)) {
      std::fputs(" <-- uninstantiated_\n", f);
    } else if (offset == offsetof(DerivedType, kindParameter_)) {
      std::fputs(" <-- kindParameter_\n", f);
    } else if (offset == offsetof(DerivedType, lenParameterKind_)) {
      std::fputs(" <-- lenParameterKind_\n", f);
    } else if (offset == offsetof(DerivedType, component_)) {
      std::fputs(" <-- component_\n", f);
    } else if (offset == offsetof(DerivedType, procPtr_)) {
      std::fputs(" <-- procPtr_\n", f);
    } else if (offset == offsetof(DerivedType, special_)) {
      std::fputs(" <-- special_\n", f);
    } else if (offset == offsetof(DerivedType, specialBitSet_)) {
      std::fputs(" <-- specialBitSet_\n", f);
    } else if (offset == offsetof(DerivedType, hasParent_)) {
      std::fputs(" <-- (flags)\n", f);
    } else {
      std::fputc('\n', f);
    }
  }
  std::fputs("  name: ", f);
  DumpScalarCharacter(f, name(), "DerivedType::name");
  const Descriptor &bindingDesc{binding()};
  std::fprintf(
      f, "\n  binding descriptor (byteSize 0x%zx): ", binding_.byteSize);
  bindingDesc.Dump(f);
  const Descriptor &compDesc{component()};
  std::fputs("\n  components:\n", f);
  if (compDesc.raw().version == CFI_VERSION &&
      compDesc.type() == TypeCode{TypeCategory::Derived, 0} &&
      compDesc.ElementBytes() == sizeof(Component) && compDesc.rank() == 1) {
    std::size_t n{compDesc.Elements()};
    for (std::size_t j{0}; j < n; ++j) {
      const Component &comp{*compDesc.ZeroBasedIndexedElement<Component>(j)};
      std::fprintf(f, "  [%3zd] ", j);
      comp.Dump(f);
    }
  } else {
    std::fputs("    bad descriptor: ", f);
    compDesc.Dump(f);
  }
  const Descriptor &specialDesc{special()};
  std::fprintf(
      f, "\n  special descriptor (byteSize 0x%zx): ", special_.byteSize);
  specialDesc.Dump(f);
  std::size_t specials{specialDesc.Elements()};
  for (std::size_t j{0}; j < specials; ++j) {
    std::fprintf(f, "  [%3zd] ", j);
    specialDesc.ZeroBasedIndexedElement<SpecialBinding>(j)->Dump(f);
  }
  return f;
}

FILE *Component::Dump(FILE *f) const {
  std::fprintf(f, "Component @ %p:\n", reinterpret_cast<const void *>(this));
  std::fputs("    name: ", f);
  DumpScalarCharacter(f, name(), "Component::name");
  if (genre_ == Genre::Data) {
    std::fputs("    Data       ", f);
  } else if (genre_ == Genre::Pointer) {
    std::fputs("    Pointer    ", f);
  } else if (genre_ == Genre::Allocatable) {
    std::fputs("    Allocatable", f);
  } else if (genre_ == Genre::Automatic) {
    std::fputs("    Automatic  ", f);
  } else {
    std::fprintf(f, "    (bad genre 0x%x)", static_cast<int>(genre_));
  }
  std::fprintf(f, " category %d  kind %d  rank %d  offset 0x%zx\n", category_,
      kind_, rank_, static_cast<std::size_t>(offset_));
  if (initialization_) {
    std::fprintf(f, " initialization @ %p:\n",
        reinterpret_cast<const void *>(initialization_));
    for (int j{0}; j < 128; j += sizeof(std::uint64_t)) {
      std::fprintf(f, " [%3d] 0x%016jx\n", j,
          static_cast<std::uintmax_t>(
              *reinterpret_cast<const std::uint64_t *>(initialization_ + j)));
    }
  }
  return f;
}

FILE *SpecialBinding::Dump(FILE *f) const {
  std::fprintf(
      f, "SpecialBinding @ %p:\n", reinterpret_cast<const void *>(this));
  switch (which_) {
  case Which::ScalarAssignment:
    std::fputs("    ScalarAssignment", f);
    break;
  case Which::ElementalAssignment:
    std::fputs("    ElementalAssignment", f);
    break;
  case Which::ReadFormatted:
    std::fputs("    ReadFormatted", f);
    break;
  case Which::ReadUnformatted:
    std::fputs("    ReadUnformatted", f);
    break;
  case Which::WriteFormatted:
    std::fputs("    WriteFormatted", f);
    break;
  case Which::WriteUnformatted:
    std::fputs("    WriteUnformatted", f);
    break;
  case Which::ElementalFinal:
    std::fputs("    ElementalFinal", f);
    break;
  case Which::AssumedRankFinal:
    std::fputs("    AssumedRankFinal", f);
    break;
  default:
    std::fprintf(f, "    rank-%d final:",
        static_cast<int>(which_) - static_cast<int>(Which::ScalarFinal));
    break;
  }
  std::fprintf(f, "    isArgDescriptorSet: 0x%x\n", isArgDescriptorSet_);
  std::fprintf(f, "    proc: %p\n", reinterpret_cast<void *>(proc_));
  return f;
}

} // namespace Fortran::runtime::typeInfo
