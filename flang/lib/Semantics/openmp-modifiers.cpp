//===-- flang/lib/Semantics/openmp-modifiers.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/openmp-modifiers.h"

#include "flang/Parser/parse-tree.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Frontend/OpenMP/OMP.h"

#include <algorithm>
#include <cassert>
#include <map>

namespace Fortran::semantics {
using namespace llvm::omp;

/// Find the highest version that exists as a key in the given map,
/// and is less than or equal to `version`.
/// Account for "version" not being a value from getOpenMPVersions().
template <typename ValueTy>
static unsigned findVersion(
    unsigned version, const std::map<unsigned, ValueTy> &map) {
  llvm::ArrayRef<unsigned> versions{llvm::omp::getOpenMPVersions()};
  assert(!versions.empty() && "getOpenMPVersions returned empty list");
  version = std::clamp(version, versions.front(), versions.back());

  // std::map is sorted with respect to keys, by default in the ascending
  // order.
  unsigned found{0};
  for (auto &[v, _] : map) {
    if (v <= version) {
      found = v;
    } else {
      break;
    }
  }

  // It can happen that the above search will not find any version, for
  // example when the minimum version in the map is higher than the current
  // version. This is really an error, but this situation should be handled
  // gracefully, so make some sensible choice and return it.
  if (found == 0) {
    found = !map.empty() ? map.begin()->first : versions.front();
  }
  return found;
}

const OmpProperties &OmpModifierDescriptor::props(unsigned version) const {
  return props_.at(findVersion(version, props_));
}

const OmpClauses &OmpModifierDescriptor::clauses(unsigned version) const {
  return clauses_.at(findVersion(version, clauses_));
}

unsigned OmpModifierDescriptor::since(llvm::omp::Clause id) const {
  unsigned found{[&]() {
    for (auto &[v, cs] : clauses_) {
      if (cs.test(id)) {
        return v;
      }
    }
    return ~0u;
  }()};

  return found <= 45 ? 0 : found;
}

// Note: The intent for these functions is to have them be automatically-
// generated in the future.

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpAlignment>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"alignment",
      /*props=*/
      {
          {45, {OmpProperty::Unique, OmpProperty::Ultimate, OmpProperty::Post}},
      },
      /*clauses=*/
      {
          {45, {Clause::OMPC_aligned}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpAlignModifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"align-modifier",
      /*props=*/
      {
          {51, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {51, {Clause::OMPC_allocate}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &
OmpGetDescriptor<parser::OmpAllocatorComplexModifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"allocator-complex-modifier",
      /*props=*/
      {
          {51, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {51, {Clause::OMPC_allocate}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &
OmpGetDescriptor<parser::OmpAllocatorSimpleModifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"allocator-simple-modifier",
      /*props=*/
      {
          {50, {OmpProperty::Exclusive, OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {50, {Clause::OMPC_allocate}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpChunkModifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"chunk-modifier",
      /*props=*/
      {
          {45, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {45, {Clause::OMPC_schedule}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpContextSelector>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"context-selector",
      /*props=*/
      {
          {50, {OmpProperty::Required, OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          // The MATCH clause takes a selector as an argument, not modifier.
          {50, {Clause::OMPC_when}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpDependenceType>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"dependence-type",
      /*props=*/
      {
          {45, {OmpProperty::Required, OmpProperty::Ultimate}},
      },
      /*clauses=*/
      {
          {45, {Clause::OMPC_depend}},
          {51, {Clause::OMPC_depend, Clause::OMPC_update}},
          {52, {Clause::OMPC_doacross}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpDeviceModifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"device-modifier",
      /*props=*/
      {
          {45, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {45, {Clause::OMPC_device}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &
OmpGetDescriptor<parser::OmpDirectiveNameModifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"directive-name-modifier",
      /*props=*/
      {
          {45, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {45, {Clause::OMPC_if}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpExpectation>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"expectation",
      /*props=*/
      {
          {51, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {51, {Clause::OMPC_from, Clause::OMPC_to}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpIterator>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"iterator",
      /*props=*/
      {
          {50, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {50, {Clause::OMPC_affinity, Clause::OMPC_depend}},
          {51,
              {Clause::OMPC_affinity, Clause::OMPC_depend, Clause::OMPC_from,
                  Clause::OMPC_map, Clause::OMPC_to}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &
OmpGetDescriptor<parser::OmpLastprivateModifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"lastprivate-modifier",
      /*props=*/
      {
          {50, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {50, {Clause::OMPC_lastprivate}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpLinearModifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"linear-modifier",
      /*props=*/
      {
          {45, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {45, {Clause::OMPC_linear}},
      },
  };
  return desc;
}

template <> //
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpMapper>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"mapper",
      /*props=*/
      {
          {50, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {50, {Clause::OMPC_from, Clause::OMPC_map, Clause::OMPC_to}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpMapType>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"map-type",
      /*props=*/
      {
          {45, {OmpProperty::Ultimate}},
      },
      /*clauses=*/
      {
          {45, {Clause::OMPC_map}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpMapTypeModifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"map-type-modifier",
      /*props=*/
      {
          {45, {}}, // Repeatable
      },
      /*clauses=*/
      {
          {45, {Clause::OMPC_map}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpOrderModifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"order-modifier",
      /*props=*/
      {
          {51, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {51, {Clause::OMPC_order}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpOrderingModifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"ordering-modifier",
      /*props=*/
      {
          {45, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {45, {Clause::OMPC_schedule}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpPrescriptiveness>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"prescriptiveness",
      /*props=*/
      {
          {51, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {51, {Clause::OMPC_grainsize, Clause::OMPC_num_tasks}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &
OmpGetDescriptor<parser::OmpReductionIdentifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"reduction-identifier",
      /*props=*/
      {
          {45, {OmpProperty::Required, OmpProperty::Ultimate}},
      },
      /*clauses=*/
      {
          {45, {Clause::OMPC_reduction}},
          {50,
              {Clause::OMPC_in_reduction, Clause::OMPC_reduction,
                  Clause::OMPC_task_reduction}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpReductionModifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"reduction-modifier",
      /*props=*/
      {
          {45, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {45, {Clause::OMPC_reduction}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &
OmpGetDescriptor<parser::OmpStepComplexModifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"step-complex-modifier",
      /*props=*/
      {
          {52, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {52, {Clause::OMPC_linear}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpStepSimpleModifier>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"step-simple-modifier",
      /*props=*/
      {
          {45, {OmpProperty::Unique, OmpProperty::Exclusive}},
      },
      /*clauses=*/
      {
          {45, {Clause::OMPC_linear}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpTaskDependenceType>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"task-dependence-type",
      /*props=*/
      {
          {45, {OmpProperty::Required, OmpProperty::Ultimate}},
      },
      /*clauses=*/
      {
          {45, {Clause::OMPC_depend}},
          {51, {Clause::OMPC_depend, Clause::OMPC_update}},
      },
  };
  return desc;
}

template <>
const OmpModifierDescriptor &OmpGetDescriptor<parser::OmpVariableCategory>() {
  static const OmpModifierDescriptor desc{
      /*name=*/"variable-category",
      /*props=*/
      {
          {45, {OmpProperty::Required, OmpProperty::Unique}},
          {50, {OmpProperty::Unique}},
      },
      /*clauses=*/
      {
          {45, {Clause::OMPC_defaultmap}},
      },
  };
  return desc;
}
} // namespace Fortran::semantics
