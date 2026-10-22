// Note: %s and %S must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl /clang:-emit-llvm /pathmap:%p=x:/path-to/ -clang:-S -clang:-o- -- %s  2>&1 | FileCheck %s --check-prefix=CHECK-PREFIX-MAP 

// CHECK-PREFIX-MAP: c"x:\\path-to\\cl-pathmap.cpp\00"

namespace std {
class source_location {
public:
struct __impl {
  const char *_M_file_name;
  const char *_M_function_name;
  unsigned _M_line;
  unsigned _M_column;
};
};
} // namespace std

const std::source_location::__impl *__m_impl = static_cast<const std::source_location::__impl *>(__builtin_source_location());
const char* file = __m_impl->_M_file_name;
