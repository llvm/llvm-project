//===-- SelectHelper.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SelectHelper_h_
#define liblldb_SelectHelper_h_

// C Includes
// C++ Includes
#include <chrono>

// Other libraries and framework includes
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"

// Project includes
#include "lldb/lldb-forward.h"

class SelectHelper {
public:
  // Defaults to infinite wait for select unless you call SetTimeout()
  SelectHelper();

  // Call SetTimeout() before calling SelectHelper::Select() to set the
  // timeout based on the current time + the timeout. This allows multiple
  // calls to SelectHelper::Select() without having to worry about the
  // absolute timeout as this class manages to set the relative timeout
  // correctly.
  void SetTimeout(const std::chrono::microseconds &timeout);

  // Call the FDSet*() functions before calling SelectHelper::Select() to
  // set the file descriptors that we will watch for when calling
  // select. This will cause FD_SET() to be called prior to calling select
  // using the "fd" provided.
  void FDSetRead(int fd);
  void FDSetWrite(int fd);
  void FDSetError(int fd);

  // Call the FDIsSet*() functions after calling SelectHelper::Select()
  // to check which file descriptors are ready for read/write/error. This
  // will contain the result of FD_ISSET after calling select for a given
  // file descriptor.
  bool FDIsSetRead(int fd) const;
  bool FDIsSetWrite(int fd) const;
  bool FDIsSetError(int fd) const;

  // Call the system's select() to wait for descriptors using
  // timeout provided in a call the SelectHelper::SetTimeout(),
  // or infinite wait if no timeout was set.
  lldb_private::Error Select();

protected:
  struct FDInfo {
    FDInfo()
        : read_set(false), write_set(false), error_set(false),
          read_is_set(false), write_is_set(false), error_is_set(false) {}

    void PrepareForSelect() {
      read_is_set = false;
      write_is_set = false;
      error_is_set = false;
    }

    bool read_set : 1, write_set : 1, error_set : 1, read_is_set : 1,
        write_is_set : 1, error_is_set : 1;
  };
  llvm::DenseMap<int, FDInfo> m_fd_map;
  llvm::Optional<std::chrono::steady_clock::time_point> m_end_time;
};

#endif // liblldb_SelectHelper_h_
