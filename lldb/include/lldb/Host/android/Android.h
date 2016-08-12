//===-- lldb-android.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_lldb_android_h_
#define LLDB_lldb_android_h_

#include <sstream>
#include <string>
#include <errno.h>

namespace std
{
	template <typename T>
	std::string to_string(T value)
	{
		std::ostringstream os ;
		os << value ;
		return os.str() ;
	}
}

#endif  // LLDB_lldb_android_h_
