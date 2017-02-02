//===-- OptionGroupPlatform.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionGroupPlatform_h_
#define liblldb_OptionGroupPlatform_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"
#include "lldb/Utility/ConstString.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// PlatformOptionGroup
//
// Make platform options available to any commands that need the settings.
//-------------------------------------------------------------------------
class OptionGroupPlatform : public OptionGroup {
public:
  OptionGroupPlatform(bool include_platform_option)
      : OptionGroup(), m_platform_name(), m_sdk_sysroot(),
        m_os_version_major(UINT32_MAX), m_os_version_minor(UINT32_MAX),
        m_os_version_update(UINT32_MAX),
        m_include_platform_option(include_platform_option) {}

  ~OptionGroupPlatform() override = default;

  llvm::ArrayRef<OptionDefinition> GetDefinitions() override;

  Error SetOptionValue(uint32_t option_idx, llvm::StringRef option_value,
                       ExecutionContext *execution_context) override;
  Error SetOptionValue(uint32_t, const char *, ExecutionContext *) = delete;

  void OptionParsingStarting(ExecutionContext *execution_context) override;

  lldb::PlatformSP CreatePlatformWithOptions(CommandInterpreter &interpreter,
                                             const ArchSpec &arch,
                                             bool make_selected, Error &error,
                                             ArchSpec &platform_arch) const;

  bool PlatformWasSpecified() const { return !m_platform_name.empty(); }

  void SetPlatformName(const char *platform_name) {
    if (platform_name && platform_name[0])
      m_platform_name.assign(platform_name);
    else
      m_platform_name.clear();
  }

  const ConstString &GetSDKRootDirectory() const { return m_sdk_sysroot; }

  void SetSDKRootDirectory(const ConstString &sdk_root_directory) {
    m_sdk_sysroot = sdk_root_directory;
  }

  const ConstString &GetSDKBuild() const { return m_sdk_build; }

  void SetSDKBuild(const ConstString &sdk_build) { m_sdk_build = sdk_build; }

  bool PlatformMatches(const lldb::PlatformSP &platform_sp) const;

protected:
  std::string m_platform_name;
  ConstString m_sdk_sysroot;
  ConstString m_sdk_build;
  uint32_t m_os_version_major;
  uint32_t m_os_version_minor;
  uint32_t m_os_version_update;
  bool m_include_platform_option;
};

} // namespace lldb_private

#endif // liblldb_OptionGroupPlatform_h_
