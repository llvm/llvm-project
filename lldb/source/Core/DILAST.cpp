//===-- DILAST.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DILAST.h"
#include "lldb/API/SBType.h"
#include "lldb/Core/ValueObjectRegister.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/RegisterContext.h"
#include "llvm/ADT/StringRef.h"

#include <vector>

namespace lldb_private {

namespace DIL {

lldb::ValueObjectSP
GetDynamicOrSyntheticValue(lldb::ValueObjectSP in_valobj_sp,
                           lldb::DynamicValueType use_dynamic,
                           bool use_synthetic) {
  Status error;

  if (!in_valobj_sp) {
    error.SetErrorString("invalid value object");
    return in_valobj_sp;
  }

  lldb::ValueObjectSP value_sp = in_valobj_sp;

  Target *target = value_sp->GetTargetSP().get();
  // If this ValueObject holds an error, then it is valuable for that.
  if (value_sp->GetError().Fail())
    return value_sp;

  if (!target)
    return lldb::ValueObjectSP();

  if (use_dynamic != lldb::eNoDynamicValues) {
    lldb::ValueObjectSP dynamic_sp = value_sp->GetDynamicValue(use_dynamic);
    if (dynamic_sp)
      value_sp = dynamic_sp;
  }

  if (use_synthetic) {
    lldb::ValueObjectSP synthetic_sp = value_sp->GetSyntheticValue();
    if (synthetic_sp)
      value_sp = synthetic_sp;
  }

  if (!value_sp)
    error.SetErrorString("invalid value object");

  return value_sp;
}

CompilerType DILASTNode::GetDereferencedResultType() const {
  auto type = result_type();
  return type.IsReferenceType() ? type.GetNonReferenceType() : type;
}

std::optional<MemberInfo>
GetFieldWithNameIndexPath(lldb::ValueObjectSP lhs_val_sp, CompilerType type,
                          const std::string &name, std::vector<uint32_t> *idx,
                          CompilerType empty_type, bool use_synthetic,
                          bool is_dynamic) {
  bool is_synthetic = false;
  // Go through the fields first.
  uint32_t num_fields = type.GetNumFields();
  lldb::ValueObjectSP empty_valobj_sp;
  for (uint32_t i = 0; i < num_fields; ++i) {
    uint64_t bit_offset = 0;
    uint32_t bitfield_bit_size = 0;
    bool is_bitfield = false;
    std::string name_sstr;
    CompilerType field_type(type.GetFieldAtIndex(
        i, name_sstr, &bit_offset, &bitfield_bit_size, &is_bitfield));
    auto field_name =
        name_sstr.length() == 0 ? std::optional<std::string>() : name_sstr;
    if (field_type.IsValid()) {
      std::optional<uint32_t> size_in_bits;
      if (is_bitfield)
        size_in_bits = bitfield_bit_size;
      struct MemberInfo field = {field_name,   field_type, size_in_bits,
                                 is_synthetic, is_dynamic, empty_valobj_sp};

      // Name can be null if this is a padding field.
      if (field.name == name) {
        if (lhs_val_sp) {
          lldb::ValueObjectSP child_valobj_sp =
              lhs_val_sp->GetChildMemberWithName(name);
          if (child_valobj_sp)
            field.val_obj_sp = child_valobj_sp;
        }

        if (idx) {
          assert(idx->empty());
          // Direct base classes are located before fields, so field members
          // needs to be offset by the number of base classes.
          idx->push_back(i + type.GetNumberOfNonEmptyBaseClasses());
        }
        return field;
      } else if (field.type.IsAnonymousType()) {
        // Every member of an anonymous struct is considered to be a member of
        // the enclosing struct or union. This applies recursively if the
        // enclosing struct or union is also anonymous.

        assert(!field.name && "Field should be unnamed.");

        std::optional<MemberInfo> field_in_anon_type =
            GetFieldWithNameIndexPath(lhs_val_sp, field.type, name, idx,
                                      empty_type, use_synthetic, is_dynamic);
        if (field_in_anon_type) {
          if (idx) {
            idx->push_back(i + type.GetNumberOfNonEmptyBaseClasses());
          }
          return field_in_anon_type.value();
        }
      }
    }
  }

  // LLDB can't access inherited fields of anonymous struct members.
  if (type.IsAnonymousType()) {
    return {};
  }

  // Go through the base classes and look for the field there.
  uint32_t num_non_empty_bases = 0;
  uint32_t num_direct_bases = type.GetNumDirectBaseClasses();
  for (uint32_t i = 0; i < num_direct_bases; ++i) {
    uint32_t bit_offset;
    auto base = type.GetDirectBaseClassAtIndex(i, &bit_offset);
    auto field = GetFieldWithNameIndexPath(
        lhs_val_sp, base, name, idx, empty_type, use_synthetic, is_dynamic);
    if (field) {
      if (idx) {
        idx->push_back(num_non_empty_bases);
      }
      return field.value();
    }
    if (base.GetNumFields() > 0) {
      num_non_empty_bases += 1;
    }
  }

  // Check for synthetic member
  if (lhs_val_sp && use_synthetic) {
    lldb::ValueObjectSP child_valobj_sp = lhs_val_sp->GetSyntheticValue();
    if (child_valobj_sp) {
      is_synthetic = true;
      uint32_t child_idx = child_valobj_sp->GetIndexOfChildWithName(name);
      child_valobj_sp = child_valobj_sp->GetChildMemberWithName(name);
      if (child_valobj_sp) {
        CompilerType field_type = child_valobj_sp->GetCompilerType();
        if (field_type.IsValid()) {
          struct MemberInfo field = {name,         field_type, {},
                                     is_synthetic, is_dynamic, child_valobj_sp};
          if (idx) {
            assert(idx->empty());
            idx->push_back(child_idx);
          }
          return field;
        }
      }
    }
  }

  if (lhs_val_sp) {
    lldb::ValueObjectSP dynamic_val_sp =
        lhs_val_sp->GetDynamicValue(lldb::eDynamicDontRunTarget);
    if (dynamic_val_sp) {
      CompilerType lhs_type = dynamic_val_sp->GetCompilerType();
      if (lhs_type.IsPointerType())
        lhs_type = lhs_type.GetPointeeType();
      is_dynamic = true;
      return GetFieldWithNameIndexPath(dynamic_val_sp, lhs_type, name, idx,
                                       empty_type, use_synthetic, is_dynamic);
    }
  }

  return {};
}

std::tuple<std::optional<MemberInfo>, std::vector<uint32_t>>
GetMemberInfo(lldb::ValueObjectSP lhs_val_sp, CompilerType type,
              const std::string &name, bool use_synthetic) {
  std::vector<uint32_t> idx;
  CompilerType empty_type;
  bool is_dynamic = false;
  std::optional<MemberInfo> member = GetFieldWithNameIndexPath(
      lhs_val_sp, type, name, &idx, empty_type, use_synthetic, is_dynamic);
  std::reverse(idx.begin(), idx.end());
  return {member, std::move(idx)};
}

static lldb::ValueObjectSP
LookupStaticIdentifier(lldb::TargetSP target_sp,
                       const llvm::StringRef &name_ref,
                       ConstString unqualified_name) {
  // List global variable with the same "basename". There can be many matches
  // from other scopes (namespaces, classes), so we do additional filtering
  // later.
  std::vector<lldb::ValueObjectSP> values;
  VariableList variable_list;
  ConstString name(name_ref);
  target_sp->GetImages().FindGlobalVariables(
      name, (size_t)std::numeric_limits<uint32_t>::max, variable_list);
  if (!variable_list.Empty()) {
    ExecutionContextScope *exe_scope = target_sp->GetProcessSP().get();
    if (exe_scope == nullptr)
      exe_scope = target_sp.get();
    for (const lldb::VariableSP &var_sp : variable_list) {
      lldb::ValueObjectSP valobj_sp(
          ValueObjectVariable::Create(exe_scope, var_sp));
      if (valobj_sp)
        values.push_back(valobj_sp);
    }
  }

  // Find the corrent variable by matching the name.
  for (uint32_t i = 0; i < values.size(); ++i) {
    lldb::ValueObjectSP val = values[i];
    if (val->GetVariable() &&
        (val->GetVariable()->NameMatches(unqualified_name) ||
         val->GetVariable()->NameMatches(ConstString(name_ref))))
      return val;
  }
  lldb::ValueObjectSP empty_obj_sp;
  return empty_obj_sp;
}

struct EnumMember {
  CompilerType type;
  ConstString name;
  llvm::APSInt value;
};

static std::vector<EnumMember> GetEnumMembers(CompilerType type) {
  std::vector<EnumMember> enum_member_list;
  if (type.IsValid()) {
    type.ForEachEnumerator(
        [&enum_member_list](const CompilerType &integer_type, ConstString name,
                            const llvm::APSInt &value) -> bool {
          EnumMember enum_member = {integer_type, name, value};
          enum_member_list.push_back(enum_member);
          return true; // Keep iterating
        });
  }
  return enum_member_list;
}

CompilerType
ResolveTypeByName(const std::string &name,
                  std::shared_ptr<ExecutionContextScope> ctx_scope) {
  // Internally types don't have global scope qualifier in their names and
  // LLDB doesn't support queries with it too.
  llvm::StringRef name_ref(name);
  bool global_scope = false;

  if (name_ref.starts_with("::")) {
    name_ref = name_ref.drop_front(2);
    global_scope = true;
  }

  std::vector<CompilerType> result_type_list;
  lldb::TargetSP target_sp = ctx_scope->CalculateTarget();
  const char *type_name = name_ref.data();
  if (type_name && type_name[0] && target_sp) {
    ModuleList &images = target_sp->GetImages();
    ConstString const_type_name(type_name);
    TypeQuery query(type_name);
    TypeResults results;
    images.FindTypes(nullptr, query, results);
    for (const lldb::TypeSP &type_sp : results.GetTypeMap().Types())
      if (type_sp)
        result_type_list.push_back(type_sp->GetFullCompilerType());

    if (auto process_sp = target_sp->GetProcessSP()) {
      for (auto *runtime : process_sp->GetLanguageRuntimes()) {
        if (auto *vendor = runtime->GetDeclVendor()) {
          auto types = vendor->FindTypes(const_type_name, UINT32_MAX);
          for (auto type : types)
            result_type_list.push_back(type);
        }
      }
    }

    if (result_type_list.empty()) {
      for (auto type_system_sp : target_sp->GetScratchTypeSystems())
        if (auto compiler_type =
                type_system_sp->GetBuiltinTypeByName(const_type_name))
          result_type_list.push_back(compiler_type);
    }
  }

  // We've found multiple types, try finding the "correct" one.
  CompilerType full_match;
  std::vector<CompilerType> partial_matches;

  for (uint32_t i = 0; i < result_type_list.size(); ++i) {
    CompilerType type = result_type_list[i];
    llvm::StringRef type_name_ref = type.GetTypeName().GetStringRef();
    ;

    if (type_name_ref == name_ref)
      full_match = type;
    else if (type_name_ref.ends_with(name_ref))
      partial_matches.push_back(type);
  }

  // Full match is always correct.
  if (full_match.IsValid())
    return full_match;

  if (!global_scope) {
    // We're looking for type, but there may be multiple candidates and which
    // one is correct may depend on the currect scope. For now just pick the
    // most "probable" type (pick a random one). TODO: Try to find a better way
    // to do this.
    if (partial_matches.size() > 0)
      return partial_matches.back();
  }

  return {};
}

static lldb::VariableSP DILFindVariable(ConstString name,
                                        VariableList *variable_list) {
  lldb::VariableSP exact_match;
  std::vector<lldb::VariableSP> possible_matches;

  typedef std::vector<lldb::VariableSP> collection;
  typedef collection::iterator iterator;

  iterator pos, end = variable_list->end();
  for (pos = variable_list->begin(); pos != end; ++pos) {
    llvm::StringRef str_ref_name = pos->get()->GetName().GetStringRef();
    // Check for global vars, which might start with '::'.
    str_ref_name.consume_front("::");

    if (str_ref_name == name.GetStringRef())
      possible_matches.push_back(*pos);
    else if (pos->get()->NameMatches(name))
      possible_matches.push_back(*pos);
  }

  auto exact_match_it =
      llvm::find_if(possible_matches, [&](lldb::VariableSP var_sp) {
        return var_sp->GetName() == name;
      });

  if (exact_match_it != llvm::adl_end(possible_matches))
    exact_match = *exact_match_it;

  if (!exact_match)
    // Look for a global var exact match.
    for (auto var_sp : possible_matches) {
      llvm::StringRef str_ref_name = var_sp->GetName().GetStringRef();
      if (str_ref_name.size() > 2 && str_ref_name[0] == ':' &&
          str_ref_name[1] == ':')
        str_ref_name = str_ref_name.drop_front(2);
      ConstString tmp_name(str_ref_name);
      if (tmp_name == name) {
        exact_match = var_sp;
        break;
      }
    }

  // Take any match at this point.
  if (!exact_match && possible_matches.size() > 0)
    exact_match = possible_matches[0];

  return exact_match;
}

std::unique_ptr<IdentifierInfo>
LookupIdentifier(const std::string &name,
                 std::shared_ptr<ExecutionContextScope> ctx_scope,
                 lldb::DynamicValueType use_dynamic, CompilerType *scope_ptr) {
  ConstString name_str(name);
  llvm::StringRef name_ref = name_str.GetStringRef();

  // Support $rax as a special syntax for accessing registers.
  // Will return an invalid value in case the requested register doesn't exist.
  if (name_ref.starts_with("$")) {
    lldb::ValueObjectSP value_sp;
    const char *reg_name = name_ref.drop_front(1).data();
    Target *target = ctx_scope->CalculateTarget().get();
    Process *process = ctx_scope->CalculateProcess().get();
    if (target && process) {
      StackFrame *stack_frame = ctx_scope->CalculateStackFrame().get();
      if (stack_frame) {
        lldb::RegisterContextSP reg_ctx(stack_frame->GetRegisterContext());
        if (reg_ctx) {
          if (const RegisterInfo *reg_info =
                  reg_ctx->GetRegisterInfoByName(reg_name))
            value_sp =
                ValueObjectRegister::Create(stack_frame, reg_ctx, reg_info);
        }
      }
    }
    return IdentifierInfo::FromValue(value_sp);
  }

  // Internally values don't have global scope qualifier in their names and
  // LLDB doesn't support queries with it too.
  bool global_scope = false;
  if (name_ref.starts_with("::")) {
    name_ref = name_ref.drop_front(2);
    global_scope = true;
  }

  // If the identifier doesn't refer to the global scope and doesn't have any
  // other scope qualifiers, try looking among the local and instance variables.
  if (!global_scope && !name_ref.contains("::")) {
    if (!scope_ptr || !scope_ptr->IsValid()) {
      // Lookup in the current frame.
      lldb::StackFrameSP frame = ctx_scope->CalculateStackFrame();
      // Try looking for a local variable in current scope.
      lldb::ValueObjectSP value_sp;
      lldb::VariableListSP var_list_sp(frame->GetInScopeVariableList(true));
      VariableList *variable_list = var_list_sp.get();
      if (variable_list) {
        lldb::VariableSP var_sp =
            DILFindVariable(ConstString(name_ref), variable_list);
        if (var_sp)
          value_sp = frame->GetValueObjectForFrameVariable(var_sp, use_dynamic);
      }
      if (!value_sp)
        value_sp = frame->FindVariable(ConstString(name_ref));

      if (value_sp)
        // Force static value, otherwise we can end up with the "real" type.
        return IdentifierInfo::FromValue(value_sp);

      // Try looking for an instance variable (class member).
      ConstString this_string("this");
      value_sp = frame->FindVariable(this_string);
      if (value_sp)
        value_sp = value_sp->GetChildMemberWithName(name_ref.data());

      if (value_sp)
        // Force static value, otherwise we can end up with the "real" type.
        return IdentifierInfo::FromValue(value_sp->GetStaticValue());

    } else {
      // In a "value" scope `this` refers to the scope object itself.
      if (name_ref == "this")
        return IdentifierInfo::FromThisKeyword(scope_ptr->GetPointerType());

      // Lookup the variable as a member of the current scope value.
      lldb::ValueObjectSP empty_sp;
      bool use_synthetic = false;
      auto [member, path] =
          GetMemberInfo(empty_sp, *scope_ptr, name_ref.data(), use_synthetic);
      if (member)
        return IdentifierInfo::FromMemberPath(member.value().type,
                                              std::move(path));
    }
  }

  // Try looking for a global or static variable.

  lldb::ValueObjectSP value;
  if (!global_scope) {
    // Try looking for static member of the current scope value, e.g.
    // `ScopeType::NAME`. NAME can include nested struct (`Nested::SUBNAME`),
    // but it cannot be part of the global scope (start with "::").
    const char *type_name = "";
    if (scope_ptr)
      type_name = scope_ptr->GetCanonicalType().GetTypeName().AsCString();
    std::string name_with_type_prefix =
        llvm::formatv("{0}::{1}", type_name, name_ref).str();
    value = LookupStaticIdentifier(ctx_scope->CalculateTarget(),
                                   name_with_type_prefix, name_str);
  }

  // Lookup a regular global variable.
  if (!value)
    value = LookupStaticIdentifier(ctx_scope->CalculateTarget(), name_ref,
                                   name_str);

  // Try looking up enum value.
  if (!value && name_ref.contains("::")) {
    auto [enum_typename, enumerator_name] = name_ref.rsplit("::");

    auto type = ResolveTypeByName(enum_typename.str(), ctx_scope);
    std::vector<EnumMember> enum_members = GetEnumMembers(type);

    for (size_t i = 0; i < enum_members.size(); i++) {
      EnumMember enum_member = enum_members[i];
      if (enum_member.name == enumerator_name) {
        uint64_t bytes = enum_member.value.getZExtValue();
        uint64_t byte_size = 0;
        if (auto temp = type.GetByteSize(ctx_scope.get()))
          byte_size = temp.value();
        lldb::TargetSP target_sp = ctx_scope->CalculateTarget();
        lldb::DataExtractorSP data_sp = std::make_shared<DataExtractor>(
            &bytes, byte_size, target_sp->GetArchitecture().GetByteOrder(),
            static_cast<uint8_t>(
                target_sp->GetArchitecture().GetAddressByteSize()));
        ExecutionContext exe_ctx(
            ExecutionContextRef(ExecutionContext(target_sp.get(), false)));
        value = ValueObject::CreateValueObjectFromData("result", *data_sp,
                                                       exe_ctx, type);
        break;
      }
    }
  }

  // Last resort, lookup as a register (e.g. `rax` or `rip`).
  if (!value) {
    Target *target = ctx_scope->CalculateTarget().get();
    Process *process = ctx_scope->CalculateProcess().get();
    if (target && process) {
      StackFrame *stack_frame = ctx_scope->CalculateStackFrame().get();
      if (stack_frame) {
        lldb::RegisterContextSP reg_ctx(stack_frame->GetRegisterContext());
        if (reg_ctx) {
          if (const RegisterInfo *reg_info =
                  reg_ctx->GetRegisterInfoByName(name_ref.data()))
            value = ValueObjectRegister::Create(stack_frame, reg_ctx, reg_info);
        }
      }
    }
  }

  // Force static value, otherwise we can end up with the "real" type.
  return IdentifierInfo::FromValue(value);
}

void ErrorNode::Accept(Visitor *v) const { v->Visit(this); }

void ScalarLiteralNode::Accept(Visitor *v) const { v->Visit(this); }

void StringLiteralNode::Accept(Visitor *v) const { v->Visit(this); }

void IdentifierNode::Accept(Visitor *v) const { v->Visit(this); }

void CStyleCastNode::Accept(Visitor *v) const { v->Visit(this); }

void MemberOfNode::Accept(Visitor *v) const { v->Visit(this); }

void ArraySubscriptNode::Accept(Visitor *v) const { v->Visit(this); }

void UnaryOpNode::Accept(Visitor *v) const { v->Visit(this); }

void SmartPtrToPtrDecay::Accept(Visitor *v) const { v->Visit(this); }

} // namespace DIL

} // namespace lldb_private
