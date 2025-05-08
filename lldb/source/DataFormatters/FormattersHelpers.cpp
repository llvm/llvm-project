//===-- FormattersHelpers.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//




#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Core/Module.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/RegularExpression.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

/* TO_UPSTREAM(BoundsSafety) ON */

/// Describes to what extent a pointer is out-of-bounds.
enum OOBKind {
  ///< Partially overlapping valid address range.
  Partial = 0,

  ///< Out-of-bounds without overlapping valid address range.
  Full,

  ///< Out-of-bounds which would overflow address space.
  Overflow,
};

struct OOBInfo {
  ///< Exclusive upper bound of OOB access.
  lldb::addr_t upper_bound;

  ///< What kind of OOB access this was.
  OOBKind kind;
};

static char const *GetOOBKindString(OOBKind kind) {
  switch (kind) {
  case OOBKind::Full:
    return "out-of-bounds";
  case OOBKind::Partial:
    return "partially out-of-bounds";
  case OOBKind::Overflow:
    return "overflown bounds";
  }
}

/// Determines whether \ref ptr (with pointee size of
/// \ref elt_size) would cause an out-of-bounds access
/// to the range [lower_bound, upper_bound). If an OOB
/// access would occur, returns OOB information, otherwise
/// returns std::nullopt.
///
/// \param[in] ptr Inclusive lower bound of pointer.
/// \param[in] upper_bound Exclusive upper bound of the
///                        address range that \ref ptr
///                        is trying to access.
/// \param[in] lower_bound Inclusive lower bound of the
///                        address range that \ref ptr
///                        is trying to access.
/// \param[in] elt_size Size of elements pointed to by \ref ptr.
///                     If 0, this function assumes the minimum
///                     bytes read by an access is 1 byte.
///
/// \returns OOBInfo describing the out-of-bounds of \ref ptr.
///          If the access was in-bounds, returns std::nullopt.
static std::optional<OOBInfo> GetOOBInfo(lldb::addr_t ptr,
                                         lldb::addr_t upper_bound,
                                         lldb::addr_t lower_bound,
                                         uint64_t elt_size) {
  if (elt_size == 0) {
    // Note when `elt_size == 0` we effectively do `ptr +1 > upper_bound`,
    // i.e. we assume that we read a single byte which is the minimum amount
    // of memory that could be accessed. This is a "best effort" attempt to
    // report OOB when we don't know the type.
    elt_size = 1;
  }

  lldb::addr_t ptr_upper_bound = ptr + elt_size;
  // We consider an access that would overflow the address space also
  // out-of-bounds.
  if (ptr_upper_bound < ptr)
    return OOBInfo{LLDB_INVALID_ADDRESS, OOBKind::Overflow};

  // Note `ptr_upper_bound == upper_bound` is ok because that corresponds
  // to accessing the last element of the buffer.

  // bounds:        lo---hi
  //    ptr: lo--hi
  //
  // or,
  //
  // bounds:     lo---hi
  //    ptr: lo--hi       <<< fully OOB since upper-bound is exclusive
  //
  if (ptr_upper_bound <= lower_bound)
    return OOBInfo{ptr_upper_bound, OOBKind::Full};

  // bounds: lo--hi
  //    ptr:        lo--hi
  //
  // or,
  //
  // bounds: lo--hi
  //    ptr:     lo--hi
  //
  if (ptr >= upper_bound)
    return OOBInfo{ptr_upper_bound, OOBKind::Full};

  // bounds:   lo--hi
  //    ptr: lo--hi
  //
  // or,
  //
  // bounds:   lo--hi
  //    ptr: lo----hi
  //
  if (ptr < lower_bound && ptr_upper_bound > lower_bound &&
      ptr_upper_bound <= upper_bound)
    return OOBInfo{ptr_upper_bound, OOBKind::Partial};

  // bounds: lo--hi
  //    ptr:   lo--hi
  //
  // or,
  //
  // bounds: lo--hi
  //    ptr: lo----hi
  //
  if (ptr >= lower_bound && ptr < upper_bound && ptr_upper_bound > upper_bound)
    return OOBInfo{ptr_upper_bound, OOBKind::Partial};

  // bounds:   lo--hi
  //    ptr: lo------hi
  //
  if (ptr < lower_bound && ptr_upper_bound > upper_bound)
    return OOBInfo{ptr_upper_bound, OOBKind::Partial};

  // Otherwise, in-bounds
  return std::nullopt;
}

/// Helper function to read a single pointer out of valobj.
static std::optional<addr_t> ReadPtr(ValueObject &valobj) {
  // Read a pointer out of valobj
  ProcessSP process_sp = valobj.GetProcessSP();
  if (!process_sp)
    return std::nullopt;
  uint32_t address_size = process_sp->GetAddressByteSize();

  DataExtractor data;
  data.SetAddressByteSize(address_size);
  Status error;
  uint32_t BytesRead = valobj.GetData(data, error);
  if (!error.Success() || BytesRead != address_size)
    return std::nullopt;

  offset_t data_offset = 0;
  return data.GetAddress(&data_offset);
}

bool lldb_private::formatters::FormatBoundsSafetyAttrPointer(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &) {
  // These types are named using the pattern:
  //   __bounds_safety::attribute_name::attribute_expression
  llvm::StringRef type_name = valobj.GetTypeName().GetStringRef();
  llvm::SmallVector<llvm::StringRef, 3> splits;
  type_name.split(splits, "::");

  if (splits.size() != 3 || splits.front() != "__bounds_safety")
    return false;

  // We need these to be null terminated for formatted print.
  std::string attribute_name = splits[1].str();
  std::string attribute_expr = splits[2].str();

  int addrsize = 12;
  if (auto target_sp = valobj.GetTargetSP())
    addrsize = target_sp->GetArchitecture().GetAddressByteSize() == 4 ? 8 : 12;

  std::optional<addr_t> maybe_ptr = ReadPtr(valobj);
  if (!maybe_ptr)
    return false;

  stream.Printf("(ptr: 0x%*.*" PRIx64 " %s: %s)", addrsize, addrsize,
                *maybe_ptr, attribute_name.c_str(), attribute_expr.c_str());
  return true;
}

bool lldb_private::formatters::FormatBoundsSafetyDynamicRangeAttrPointer(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &) {
  // These types are named using the pattern:
  //   __bounds_safety::dynamic_range::start_expr::end_expr
  llvm::StringRef type_name = valobj.GetTypeName().GetStringRef();
  llvm::SmallVector<llvm::StringRef, 4> splits;
  type_name.split(splits, "::");

  if (splits.size() != 4 || splits[0] != "__bounds_safety" ||
      splits[1] != "dynamic_range") {
    auto str = type_name.str();
    stream.Printf("%s", str.c_str());
    return true;
  }

  std::string start_expr = splits[2].str();
  std::string end_expr = splits[3].str();

  int addrsize = 12;
  if (auto target_sp = valobj.GetTargetSP())
    addrsize = target_sp->GetArchitecture().GetAddressByteSize() == 4 ? 8 : 12;

  std::optional<addr_t> maybe_ptr = ReadPtr(valobj);
  if (!maybe_ptr)
    return false;

  stream.Printf("(ptr: 0x%*.*" PRIx64, addrsize, addrsize, *maybe_ptr);
  if (!start_expr.empty())
    stream.Printf(" start_expr: %s", start_expr.c_str());
  if (!end_expr.empty())
    stream.Printf(" end_expr: %s", end_expr.c_str());
  stream.Printf(")");
  return true;
}

bool lldb_private::formatters::FormatBoundsSafetyPointer(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &) {
  ProcessSP process_sp = valobj.GetProcessSP();
  if (!process_sp)
    return false;

  CompilerType compiler_type = valobj.GetCompilerType();
  if (!compiler_type)
    return false;

  const bool is_bidi_indexable = compiler_type.IsBoundsSafetyBidiIndexable();
  const bool is_indexable = compiler_type.IsBoundsSafetyIndexable();
  if (!is_indexable && !is_bidi_indexable)
    return false;

  // BoundsSafety pointers are wide pointers e.g.
  //
  // struct wide_ptr {
  //  T *raw_ptr;
  //  T *upper_bound; // Exists in both indexable and bidi indexable case
  //  T *lower_bound; // Exists only for bidi indexable case
  // };
  //
  // We want to extract the upper bound, and lower bound if it exists.

  uint32_t address_size = process_sp->GetAddressByteSize();
  uint32_t expected_data_size =
      is_bidi_indexable ? 3 * address_size : 2 * address_size;

  DataExtractor data;
  data.SetAddressByteSize(address_size);
  Status error;
  uint32_t BytesRead = valobj.GetData(data, error);
  if (!error.Success() || BytesRead != expected_data_size)
    return false;

  offset_t data_offset = 0;
  auto read_next_pointer = [&data, &data_offset]() {
    return addr_t(data.GetAddress(&data_offset));
  };

  addr_t ptr = read_next_pointer();
  addr_t upper_bound = read_next_pointer();
  addr_t lower_bound = is_bidi_indexable ? read_next_pointer() : ptr;

  int addrsize = 12;
  if (auto target_sp = valobj.GetTargetSP())
    addrsize = target_sp->GetArchitecture().GetAddressByteSize() == 4 ? 8 : 12;

  // Compute size of pointee if it is known.
  CompilerType pointee_type;
  uint64_t pointee_byte_size = 0;
  if (compiler_type.IsPointerType(&pointee_type)) {
    pointee_byte_size =
        llvm::expectedToOptional(
            pointee_type.GetByteSize(valobj.GetTargetSP().get()))
            .value_or(0);
  }

  if (auto maybe_oob_info = GetOOBInfo(ptr, /*upper_bound=*/upper_bound,
                                       /*lower_bound=*/lower_bound,
                                       /*elt_size=*/pointee_byte_size)) {
    // For OOB accesses, we print ptr's upper bound.
    stream.Printf("(%s ptr: 0x%*.*" PRIx64 "..0x%*.*" PRIx64,
                  GetOOBKindString(maybe_oob_info->kind), addrsize, addrsize,
                  ptr, addrsize, addrsize, maybe_oob_info->upper_bound);
  } else {
    stream.Printf("(ptr: 0x%*.*" PRIx64, addrsize, addrsize, ptr);
  }

  if (is_bidi_indexable) {
    stream.Printf(", bounds: 0x%*.*" PRIx64 "..0x%*.*" PRIx64 ")", addrsize,
                  addrsize, lower_bound, addrsize, addrsize, upper_bound);
  } else {
    stream.Printf(", upper bound: 0x%*.*" PRIx64 ")", addrsize, addrsize,
                  upper_bound);
  }

  // After we apply the BoundsSafety formatting for the pointer if the base type
  // e.g. char* has a summary we need to obtain that summary and append it.

  // We need to strip the BoundsSafety attributes and create a new ValueObject
  // for that stripped type.
  compiler_type = compiler_type.AddBoundsSafetyUnspecifiedAttribute();
  lldb::ValueObjectSP bounds_safety_attributes_removed = valobj.Cast(compiler_type);

  const char *base_summary = bounds_safety_attributes_removed->GetSummaryAsCString();
  if (base_summary)
    stream.Printf(" %s", base_summary);

  return true;
}
/* TO_UPSTREAM(BoundsSafety) OFF */

void lldb_private::formatters::AddFormat(
    TypeCategoryImpl::SharedPointer category_sp, lldb::Format format,
    llvm::StringRef type_name, TypeFormatImpl::Flags flags, bool regex) {
  lldb::TypeFormatImplSP format_sp(new TypeFormatImpl_Format(format, flags));

  FormatterMatchType match_type =
      regex ? eFormatterMatchRegex : eFormatterMatchExact;
  category_sp->AddTypeFormat(type_name, match_type, format_sp);
}

void lldb_private::formatters::AddSummary(
    TypeCategoryImpl::SharedPointer category_sp, TypeSummaryImplSP summary_sp,
    llvm::StringRef type_name, bool regex) {
  FormatterMatchType match_type =
      regex ? eFormatterMatchRegex : eFormatterMatchExact;
  category_sp->AddTypeSummary(type_name, match_type, summary_sp);
}

void lldb_private::formatters::AddStringSummary(
    TypeCategoryImpl::SharedPointer category_sp, const char *string,
    llvm::StringRef type_name, TypeSummaryImpl::Flags flags, bool regex) {
  lldb::TypeSummaryImplSP summary_sp(new StringSummaryFormat(flags, string));

  FormatterMatchType match_type =
      regex ? eFormatterMatchRegex : eFormatterMatchExact;
  category_sp->AddTypeSummary(type_name, match_type, summary_sp);
}

void lldb_private::formatters::AddOneLineSummary(
    TypeCategoryImpl::SharedPointer category_sp, llvm::StringRef type_name,
    TypeSummaryImpl::Flags flags, bool regex) {
  flags.SetShowMembersOneLiner(true);
  lldb::TypeSummaryImplSP summary_sp(new StringSummaryFormat(flags, ""));

  FormatterMatchType match_type =
      regex ? eFormatterMatchRegex : eFormatterMatchExact;
  category_sp->AddTypeSummary(type_name, match_type, summary_sp);
}

void lldb_private::formatters::AddCXXSummary(
    TypeCategoryImpl::SharedPointer category_sp,
    CXXFunctionSummaryFormat::Callback funct, const char *description,
    llvm::StringRef type_name, TypeSummaryImpl::Flags flags, bool regex) {
  lldb::TypeSummaryImplSP summary_sp(
      new CXXFunctionSummaryFormat(flags, funct, description));

  FormatterMatchType match_type =
      regex ? eFormatterMatchRegex : eFormatterMatchExact;
  category_sp->AddTypeSummary(type_name, match_type, summary_sp);
}

void lldb_private::formatters::AddCXXSynthetic(
    TypeCategoryImpl::SharedPointer category_sp,
    CXXSyntheticChildren::CreateFrontEndCallback generator,
    const char *description, llvm::StringRef type_name,
    ScriptedSyntheticChildren::Flags flags, bool regex) {
  lldb::SyntheticChildrenSP synth_sp(
      new CXXSyntheticChildren(flags, description, generator));
  FormatterMatchType match_type =
      regex ? eFormatterMatchRegex : eFormatterMatchExact;
  category_sp->AddTypeSynthetic(type_name, match_type, synth_sp);
}

void lldb_private::formatters::AddFilter(
    TypeCategoryImpl::SharedPointer category_sp,
    std::vector<std::string> children, const char *description,
    llvm::StringRef type_name, ScriptedSyntheticChildren::Flags flags,
    bool regex) {
  TypeFilterImplSP filter_sp(new TypeFilterImpl(flags));
  for (auto child : children)
    filter_sp->AddExpressionPath(child);
  FormatterMatchType match_type =
      regex ? eFormatterMatchRegex : eFormatterMatchExact;
  category_sp->AddTypeFilter(type_name, match_type, filter_sp);
}

std::optional<size_t>
lldb_private::formatters::ExtractIndexFromString(const char *item_name) {
  if (!item_name || !*item_name)
    return std::nullopt;
  if (*item_name != '[')
    return std::nullopt;
  item_name++;
  char *endptr = nullptr;
  unsigned long int idx = ::strtoul(item_name, &endptr, 0);
  if ((idx == 0 && endptr == item_name) || idx == ULONG_MAX)
    return std::nullopt;
  return idx;
}

Address
lldb_private::formatters::GetArrayAddressOrPointerValue(ValueObject &valobj) {
  lldb::addr_t data_addr = LLDB_INVALID_ADDRESS;
  AddressType type;

  if (valobj.IsPointerType())
    data_addr = valobj.GetPointerValue(&type);
  else if (valobj.IsArrayType())
    data_addr = valobj.GetAddressOf(/*scalar_is_load_address=*/true, &type);
  if (data_addr != LLDB_INVALID_ADDRESS && type == eAddressTypeFile)
    return Address(data_addr, valobj.GetModule()->GetSectionList());

  return data_addr;
}
