//===-- NativeRegisterContextRegisterInfo.h ----------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_NativeRegisterContextRegisterInfo_h
#define lldb_NativeRegisterContextRegisterInfo_h

#include <memory>

#include "NativeRegisterContext.h"
#include "Plugins/Process/Utility/RegisterInfoInterface.h"

namespace lldb_private {
class NativeRegisterContextRegisterInfo : public NativeRegisterContext {
public:
  ///
  /// Construct a NativeRegisterContextRegisterInfo, taking ownership
  /// of the register_info_interface pointer.
  ///
  NativeRegisterContextRegisterInfo(
      NativeThreadProtocol &thread, uint32_t concrete_frame_idx,
      RegisterInfoInterface *register_info_interface);

  uint32_t GetRegisterCount() const override;

  uint32_t GetUserRegisterCount() const override;

  const RegisterInfo *GetRegisterInfoAtIndex(uint32_t reg_index) const override;

  const RegisterInfoInterface &GetRegisterInfoInterface() const;

private:
  std::unique_ptr<RegisterInfoInterface> m_register_info_interface_up;
};
}
#endif
