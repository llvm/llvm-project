//===-- UnixSignals.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_UnixSignals_h_
#define lldb_UnixSignals_h_

// C Includes
// C++ Includes
#include <map>
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class UnixSignals {
public:
  static lldb::UnixSignalsSP Create(const ArchSpec &arch);

  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  UnixSignals();

  virtual ~UnixSignals();

  const char *GetSignalAsCString(int32_t signo) const;

  bool SignalIsValid(int32_t signo) const;

  int32_t GetSignalNumberFromName(const char *name) const;

  const char *GetSignalInfo(int32_t signo, bool &should_suppress,
                            bool &should_stop, bool &should_notify) const;

  bool GetShouldSuppress(int32_t signo) const;

  bool SetShouldSuppress(int32_t signo, bool value);

  bool SetShouldSuppress(const char *signal_name, bool value);

  bool GetShouldStop(int32_t signo) const;

  bool SetShouldStop(int32_t signo, bool value);
  bool SetShouldStop(const char *signal_name, bool value);

  bool GetShouldNotify(int32_t signo) const;

  bool SetShouldNotify(int32_t signo, bool value);

  bool SetShouldNotify(const char *signal_name, bool value);

  // These provide an iterator through the signals available on this system.
  // Call GetFirstSignalNumber to get the first entry, then iterate on
  // GetNextSignalNumber
  // till you get back LLDB_INVALID_SIGNAL_NUMBER.
  int32_t GetFirstSignalNumber() const;

  int32_t GetNextSignalNumber(int32_t current_signal) const;

  int32_t GetNumSignals() const;

  int32_t GetSignalAtIndex(int32_t index) const;

  ConstString GetShortName(ConstString name) const;

  // We assume that the elements of this object are constant once it is
  // constructed,
  // since a process should never need to add or remove symbols as it runs.  So
  // don't
  // call these functions anywhere but the constructor of your subclass of
  // UnixSignals or in
  // your Process Plugin's GetUnixSignals method before you return the
  // UnixSignal object.

  void AddSignal(int signo, const char *name, bool default_suppress,
                 bool default_stop, bool default_notify,
                 const char *description, const char *alias = nullptr);

  void RemoveSignal(int signo);

protected:
  //------------------------------------------------------------------
  // Classes that inherit from UnixSignals can see and modify these
  //------------------------------------------------------------------

  struct Signal {
    ConstString m_name;
    ConstString m_alias;
    std::string m_description;
    bool m_suppress : 1, m_stop : 1, m_notify : 1;

    Signal(const char *name, bool default_suppress, bool default_stop,
           bool default_notify, const char *description, const char *alias);

    ~Signal() {}
  };

  virtual void Reset();

  typedef std::map<int32_t, Signal> collection;

  collection m_signals;

  // GDBRemote signals need to be copyable.
  UnixSignals(const UnixSignals &rhs);

  const UnixSignals &operator=(const UnixSignals &rhs) = delete;
};

} // Namespace lldb
#endif // lldb_UnixSignals_h_
