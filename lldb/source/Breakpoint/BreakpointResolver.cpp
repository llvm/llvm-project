//===-- BreakpointResolver.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/BreakpointResolver.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
// Have to include the other breakpoint resolver types here so the static
// create from StructuredData can call them.
#include "lldb/Breakpoint/BreakpointResolverAddress.h"
#include "lldb/Breakpoint/BreakpointResolverFileLine.h"
#include "lldb/Breakpoint/BreakpointResolverFileRegex.h"
#include "lldb/Breakpoint/BreakpointResolverName.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb_private;
using namespace lldb;

//----------------------------------------------------------------------
// BreakpointResolver:
//----------------------------------------------------------------------
const char *BreakpointResolver::g_ty_to_name[] = {"FileAndLine", "Address",
                                                  "SymbolName",  "SourceRegex",
                                                  "Exception",   "Unknown"};

const char *BreakpointResolver::g_option_names[static_cast<uint32_t>(
    BreakpointResolver::OptionNames::LastOptionName)] = {
    "AddressOffset", "Exact",        "FileName",   "Inlines", "Language",
    "LineNumber",    "ModuleName",   "NameMask",   "Offset",  "Regex",
    "SectionName",   "SkipPrologue", "SymbolNames"};

const char *BreakpointResolver::ResolverTyToName(enum ResolverTy type) {
  if (type > LastKnownResolverType)
    return g_ty_to_name[UnknownResolver];

  return g_ty_to_name[type];
}

BreakpointResolver::ResolverTy
BreakpointResolver::NameToResolverTy(llvm::StringRef name) {
  for (size_t i = 0; i < LastKnownResolverType; i++) {
    if (name == g_ty_to_name[i])
      return (ResolverTy)i;
  }
  return UnknownResolver;
}

BreakpointResolver::BreakpointResolver(Breakpoint *bkpt,
                                       const unsigned char resolverTy,
                                       lldb::addr_t offset)
    : m_breakpoint(bkpt), m_offset(offset), SubclassID(resolverTy) {}

BreakpointResolver::~BreakpointResolver() {}

BreakpointResolverSP BreakpointResolver::CreateFromStructuredData(
    const StructuredData::Dictionary &resolver_dict, Status &error) {
  BreakpointResolverSP result_sp;
  if (!resolver_dict.IsValid()) {
    error.SetErrorString("Can't deserialize from an invalid data object.");
    return result_sp;
  }

  llvm::StringRef subclass_name;

  bool success = resolver_dict.GetValueForKeyAsString(
      GetSerializationSubclassKey(), subclass_name);

  if (!success) {
    error.SetErrorStringWithFormat(
        "Resolver data missing subclass resolver key");
    return result_sp;
  }

  ResolverTy resolver_type = NameToResolverTy(subclass_name);
  if (resolver_type == UnknownResolver) {
    error.SetErrorStringWithFormatv("Unknown resolver type: {0}.",
                                    subclass_name);
    return result_sp;
  }

  StructuredData::Dictionary *subclass_options = nullptr;
  success = resolver_dict.GetValueForKeyAsDictionary(
      GetSerializationSubclassOptionsKey(), subclass_options);
  if (!success || !subclass_options || !subclass_options->IsValid()) {
    error.SetErrorString("Resolver data missing subclass options key.");
    return result_sp;
  }

  lldb::addr_t offset;
  success = subclass_options->GetValueForKeyAsInteger(
      GetKey(OptionNames::Offset), offset);
  if (!success) {
    error.SetErrorString("Resolver data missing offset options key.");
    return result_sp;
  }

  BreakpointResolver *resolver;

  switch (resolver_type) {
  case FileLineResolver:
    resolver = BreakpointResolverFileLine::CreateFromStructuredData(
        nullptr, *subclass_options, error);
    break;
  case AddressResolver:
    resolver = BreakpointResolverAddress::CreateFromStructuredData(
        nullptr, *subclass_options, error);
    break;
  case NameResolver:
    resolver = BreakpointResolverName::CreateFromStructuredData(
        nullptr, *subclass_options, error);
    break;
  case FileRegexResolver:
    resolver = BreakpointResolverFileRegex::CreateFromStructuredData(
        nullptr, *subclass_options, error);
    break;
  case ExceptionResolver:
    error.SetErrorString("Exception resolvers are hard.");
    break;
  default:
    llvm_unreachable("Should never get an unresolvable resolver type.");
  }

  if (!error.Success()) {
    return result_sp;
  } else {
    // Add on the global offset option:
    resolver->SetOffset(offset);
    return BreakpointResolverSP(resolver);
  }
}

StructuredData::DictionarySP BreakpointResolver::WrapOptionsDict(
    StructuredData::DictionarySP options_dict_sp) {
  if (!options_dict_sp || !options_dict_sp->IsValid())
    return StructuredData::DictionarySP();

  StructuredData::DictionarySP type_dict_sp(new StructuredData::Dictionary());
  type_dict_sp->AddStringItem(GetSerializationSubclassKey(), GetResolverName());
  type_dict_sp->AddItem(GetSerializationSubclassOptionsKey(), options_dict_sp);

  // Add the m_offset to the dictionary:
  options_dict_sp->AddIntegerItem(GetKey(OptionNames::Offset), m_offset);

  return type_dict_sp;
}

void BreakpointResolver::SetBreakpoint(Breakpoint *bkpt) {
  m_breakpoint = bkpt;
}

void BreakpointResolver::ResolveBreakpointInModules(SearchFilter &filter,
                                                    ModuleList &modules) {
  filter.SearchInModuleList(*this, modules);
}

void BreakpointResolver::ResolveBreakpoint(SearchFilter &filter) {
  filter.Search(*this);
}

void BreakpointResolver::SetSCMatchesByLine(SearchFilter &filter,
                                            SymbolContextList &sc_list,
                                            bool skip_prologue,
                                            llvm::StringRef log_ident) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));

  while (sc_list.GetSize() > 0) {
    SymbolContextList tmp_sc_list;
    unsigned current_idx = 0;
    SymbolContext sc;
    bool first_entry = true;

    FileSpec match_file_spec;
    FileSpec match_original_file_spec;
    uint32_t closest_line_number = UINT32_MAX;

    // Pull out the first entry, and all the others that match its file spec,
    // and stuff them in the tmp list.
    while (current_idx < sc_list.GetSize()) {
      bool matches;

      sc_list.GetContextAtIndex(current_idx, sc);
      if (first_entry) {
        match_file_spec = sc.line_entry.file;
        match_original_file_spec = sc.line_entry.original_file;
        matches = true;
        first_entry = false;
      } else
        matches = ((sc.line_entry.file == match_file_spec) ||
                   (sc.line_entry.original_file == match_original_file_spec));

      if (matches) {
        tmp_sc_list.Append(sc);
        sc_list.RemoveContextAtIndex(current_idx);

        // ResolveSymbolContext will always return a number that is >= the line
        // number you pass in. So the smaller line number is always better.
        if (sc.line_entry.line < closest_line_number)
          closest_line_number = sc.line_entry.line;
      } else
        current_idx++;
    }

    // Okay, we've found the closest line number match, now throw away all the
    // others:

    current_idx = 0;
    while (current_idx < tmp_sc_list.GetSize()) {
      if (tmp_sc_list.GetContextAtIndex(current_idx, sc)) {
        if (sc.line_entry.line != closest_line_number)
          tmp_sc_list.RemoveContextAtIndex(current_idx);
        else
          current_idx++;
      }
    }

    // Next go through and see if there are line table entries that are
    // contiguous, and if so keep only the first of the contiguous range:

    current_idx = 0;
    std::map<Block *, lldb::addr_t> blocks_with_breakpoints;

    while (current_idx < tmp_sc_list.GetSize()) {
      if (tmp_sc_list.GetContextAtIndex(current_idx, sc)) {
        if (blocks_with_breakpoints.find(sc.block) !=
            blocks_with_breakpoints.end())
          tmp_sc_list.RemoveContextAtIndex(current_idx);
        else {
          blocks_with_breakpoints.insert(std::pair<Block *, lldb::addr_t>(
              sc.block, sc.line_entry.range.GetBaseAddress().GetFileAddress()));
          current_idx++;
        }
      }
    }

    // and make breakpoints out of the closest line number match.

    uint32_t tmp_sc_list_size = tmp_sc_list.GetSize();

    for (uint32_t i = 0; i < tmp_sc_list_size; i++) {
      if (tmp_sc_list.GetContextAtIndex(i, sc)) {
        Address line_start = sc.line_entry.range.GetBaseAddress();
        if (line_start.IsValid()) {
          if (filter.AddressPasses(line_start)) {
            // If the line number is before the prologue end, move it there...
            bool skipped_prologue = false;
            if (skip_prologue) {
              if (sc.function) {
                Address prologue_addr(
                    sc.function->GetAddressRange().GetBaseAddress());
                if (prologue_addr.IsValid() && (line_start == prologue_addr)) {
                  const uint32_t prologue_byte_size =
                      sc.function->GetPrologueByteSize();
                  if (prologue_byte_size) {
                    prologue_addr.Slide(prologue_byte_size);

                    if (filter.AddressPasses(prologue_addr)) {
                      skipped_prologue = true;
                      line_start = prologue_addr;
                    }
                  }
                }
              }
            }

            BreakpointLocationSP bp_loc_sp(AddLocation(line_start));
            if (log && bp_loc_sp && !m_breakpoint->IsInternal()) {
              StreamString s;
              bp_loc_sp->GetDescription(&s, lldb::eDescriptionLevelVerbose);
              log->Printf("Added location (skipped prologue: %s): %s \n",
                          skipped_prologue ? "yes" : "no", s.GetData());
            }
          } else if (log) {
            log->Printf("Breakpoint %s at file address 0x%" PRIx64
                        " didn't pass the filter.\n",
                        log_ident.str().c_str(), line_start.GetFileAddress());
          }
        } else {
          if (log)
            log->Printf(
                "error: Unable to set breakpoint %s at file address 0x%" PRIx64
                "\n",
                log_ident.str().c_str(), line_start.GetFileAddress());
        }
      }
    }
  }
}

BreakpointLocationSP BreakpointResolver::AddLocation(Address loc_addr,
                                                     bool *new_location) {
  loc_addr.Slide(m_offset);
  return m_breakpoint->AddLocation(loc_addr, new_location);
}

void BreakpointResolver::SetOffset(lldb::addr_t offset) {
  // There may already be an offset, so we are actually adjusting location
  // addresses by the difference.
  // lldb::addr_t slide = offset - m_offset;
  // FIXME: We should go fix up all the already set locations for the new slide.

  m_offset = offset;
}
