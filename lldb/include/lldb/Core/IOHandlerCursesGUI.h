//===-- IOHandlerCursesGUI.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IOHandlerCursesGUI_h_
#define liblldb_IOHandlerCursesGUI_h_

#include "lldb/Core/IOHandler.h"

namespace lldb_private {

class IOHandlerCursesGUI : public IOHandler {
public:
  IOHandlerCursesGUI(Debugger &debugger);

  ~IOHandlerCursesGUI() override;

  void Run() override;

  void Cancel() override;

  bool Interrupt() override;

  void GotEOF() override;

  void Activate() override;

  void Deactivate() override;

protected:
  curses::ApplicationAP m_app_ap;
};

} // namespace lldb_private

#endif // liblldb_IOHandlerCursesGUI_h_
