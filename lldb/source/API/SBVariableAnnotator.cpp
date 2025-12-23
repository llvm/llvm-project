//===-- SBVariableAnnotator.cpp
//-------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBVariableAnnotator.h"
#include "SBVariableAnnotator.h"
#include "lldb/API/SBInstruction.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/Core/Disassembler.h" // containts VariableAnnotator declaration
#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Utility/Instrumentation.h"

using namespace lldb;
using namespace lldb_private;

SBVariableAnnotator::SBVariableAnnotator() { LLDB_INSTRUMENT_VA(this); }

SBVariableAnnotator::SBVariableAnnotator(const SBVariableAnnotator &rhs) {
  LLDB_INSTRUMENT_VA(this);
}

const SBVariableAnnotator &
SBVariableAnnotator::operator=(const SBVariableAnnotator &rhs) {
  LLDB_INSTRUMENT_VA(this);

  //   if (this != &rhs)
  //     // TODO implement
  return *this;
}

SBVariableAnnotator::~SBVariableAnnotator() = default;

SBVariableAnnotator::operator bool() const {
  LLDB_INSTRUMENT_VA(this);

  return m_opaque_sp.get() != nullptr;
}

bool lldb::SBVariableAnnotator::IsValid() const {
  LLDB_INSTRUMENT_VA(this);
  return this->operator bool();
}

lldb::SBStructuredData
lldb::SBVariableAnnotator::AnnotateStructured(SBInstruction inst) {
  LLDB_INSTRUMENT_VA(this, inst);

  lldb::SBStructuredData result;

  if (lldb::VariableAnnotatorSP annotator_sp = GetSP())
    if (lldb::InstructionSP inst_sp = inst.GetOpaque()) {
      auto array_sp = StructuredData::ArraySP();

      const std::vector<lldb_private::VariableAnnotation>
          structured_annotations = annotator_sp->AnnotateStructured(*inst_sp);

      for (const VariableAnnotation &annotation : structured_annotations) {
        auto dict_sp = std::make_shared<StructuredData::Dictionary>();

        dict_sp->AddStringItem("variable_name", annotation.variable_name);
        dict_sp->AddStringItem("location_description",
                               annotation.location_description);
        dict_sp->AddBooleanItem("is_live", annotation.is_live);
        if (annotation.address_range.has_value()) {
          const auto &range = *annotation.address_range;
          dict_sp->AddItem("start_address",
                           std::make_shared<StructuredData::UnsignedInteger>(
                               range.GetBaseAddress().GetFileAddress()));
          dict_sp->AddItem("end_address",
                           std::make_shared<StructuredData::UnsignedInteger>(
                               range.GetBaseAddress().GetFileAddress() +
                               range.GetByteSize()));
        }
        dict_sp->AddItem("register_kind",
                         std::make_shared<StructuredData::UnsignedInteger>(
                             annotation.register_kind));
        if (annotation.decl_file.has_value())
          dict_sp->AddStringItem("decl_file", *annotation.decl_file);
        if (annotation.decl_line.has_value())
          dict_sp->AddItem("decl_line",
                           std::make_shared<StructuredData::UnsignedInteger>(
                               *annotation.decl_line));
        if (annotation.type_name.has_value())
          dict_sp->AddStringItem("type_name", *annotation.type_name);

        array_sp->AddItem(dict_sp);
      }

      result.m_impl_up->SetObjectSP(array_sp);
    }
  return result;
}

lldb::SBVariableAnnotator::SBVariableAnnotator(
    const lldb::VariableAnnotatorSP &annotator_sp)
    : m_opaque_sp(annotator_sp) {
  LLDB_INSTRUMENT_VA(this, annotator_sp);
}

lldb::VariableAnnotatorSP lldb::SBVariableAnnotator::GetSP() const {
  LLDB_INSTRUMENT_VA(this);
  return m_opaque_sp;
}

void lldb::SBVariableAnnotator::SetSP(
    const lldb::VariableAnnotatorSP &annotator_sp) {
  LLDB_INSTRUMENT_VA(this, annotator_sp);
  m_opaque_sp = annotator_sp;
}
