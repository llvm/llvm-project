//===-- Baton.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Baton_h_
#define lldb_Baton_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Baton Baton.h "lldb/Core/Baton.h"
/// @brief A class designed to wrap callback batons so they can cleanup
///        any acquired resources
///
/// This class is designed to be used by any objects that have a
/// callback function that takes a baton where the baton might need to
/// free/delete/close itself.
///
/// The default behavior is to not free anything. Subclasses can
/// free any needed resources in their destructors.
//----------------------------------------------------------------------
class Baton {
public:
  explicit Baton(void *p) : m_data(p) {}

  virtual ~Baton() {
    // The default destructor for a baton does NOT attempt to clean up
    // anything in m_baton
  }

  virtual void GetDescription(Stream *s, lldb::DescriptionLevel level) const;

  void *m_data; // Leave baton public for easy access

private:
  //------------------------------------------------------------------
  // For Baton only
  //------------------------------------------------------------------
  DISALLOW_COPY_AND_ASSIGN(Baton);
};

} // namespace lldb_private

#endif // lldb_Baton_h_
