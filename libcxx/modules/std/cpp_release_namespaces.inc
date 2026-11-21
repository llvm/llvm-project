// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// PROTOTYPE: P3866R0 standard-release namespaces.
//
// The using-directives in the release namespaces come from the global module fragment, and they are not visible to
// importers of the std module. Without re-exporting them here, only the names explicitly exported in a release
// namespace (e.g. std::cpp29::vector, see vector.inc) would be usable, and std::cpp29::string would be ill-formed.
//
// Clang accepts exporting using-directives from an exported namespace definition. The alternative would be to list
// every single name of the library in every single release namespace.
export namespace std {
#if _LIBCPP_STD_VER >= 11
  namespace cpp11 {
    using namespace std;
  }
#endif
#if _LIBCPP_STD_VER >= 14
  namespace cpp14 {
    using namespace std;
  }
#endif
#if _LIBCPP_STD_VER >= 17
  namespace cpp17 {
    using namespace std;
  }
#endif
#if _LIBCPP_STD_VER >= 20
  namespace cpp20 {
    using namespace std;
  }
#endif
#if _LIBCPP_STD_VER >= 23
  namespace cpp23 {
    using namespace std;
  }
#endif
#if _LIBCPP_STD_VER >= 26
  namespace cpp26 {
    using namespace std;
  }
#endif
#if _LIBCPP_STD_VER >= 29
  namespace cpp29 {
    using namespace std;

    namespace pmr {
      using namespace std::pmr;
    }
  }
#endif
} // namespace std
