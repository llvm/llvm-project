//===-- LanguageCategory.h----------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_LanguageCategory_h_
#define lldb_LanguageCategory_h_


#include "lldb/DataFormatters/FormatCache.h"
#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/lldb-public.h"

#include <memory>

namespace lldb_private {

class LanguageCategory {
public:
  typedef std::unique_ptr<LanguageCategory> UniquePointer;

  LanguageCategory(lldb::LanguageType lang_type);

  bool Get(FormattersMatchData &match_data, lldb::TypeFormatImplSP &format_sp);

  bool Get(FormattersMatchData &match_data, lldb::TypeSummaryImplSP &format_sp);

  bool Get(FormattersMatchData &match_data,
           lldb::SyntheticChildrenSP &format_sp);

  bool Get(FormattersMatchData &match_data,
           lldb::TypeValidatorImplSP &format_sp);

  bool GetHardcoded(FormatManager &fmt_mgr, FormattersMatchData &match_data,
                    lldb::TypeFormatImplSP &format_sp);

  bool GetHardcoded(FormatManager &fmt_mgr, FormattersMatchData &match_data,
                    lldb::TypeSummaryImplSP &format_sp);

  bool GetHardcoded(FormatManager &fmt_mgr, FormattersMatchData &match_data,
                    lldb::SyntheticChildrenSP &format_sp);

  bool GetHardcoded(FormatManager &fmt_mgr, FormattersMatchData &match_data,
                    lldb::TypeValidatorImplSP &format_sp);

  lldb::TypeCategoryImplSP GetCategory() const;

  FormatCache &GetFormatCache();

  void Enable();

  void Disable();

  bool IsEnabled();

private:
  lldb::TypeCategoryImplSP m_category_sp;

  HardcodedFormatters::HardcodedFormatFinder m_hardcoded_formats;
  HardcodedFormatters::HardcodedSummaryFinder m_hardcoded_summaries;
  HardcodedFormatters::HardcodedSyntheticFinder m_hardcoded_synthetics;
  HardcodedFormatters::HardcodedValidatorFinder m_hardcoded_validators;

  lldb_private::FormatCache m_format_cache;

  bool m_enabled;
};

} // namespace lldb_private

#endif // lldb_LanguageCategory_h_
