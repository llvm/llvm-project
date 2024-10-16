Files in this directory are used by Flang (the compiler) and FortranRuntime
(the runtime library for Flang-compiled programs). They must be compatible by
both. For definitions used only by Flang, consider
`flang/{lib,include/flang}/Support` instead. For definitions used only by
the runtime, consider `flang-rt/{lib,include/flang-rt}/flang_rt`.

The requirements for common code include:

 * No dependence to LLVM, including LLVMSupport.

 * No link-dependence to the C++ runtime. This means that e.g. `std::string`
   cannot be used.

 * No use of `std::optional<T>`; `fortran::common::optional<T>` can be used
   instead.

 * Preprocessor macros from `config.h` and CMake
   `(target/add)_compile_definitions` must be defined by both build scripts.
   See `flang/cmake/modules/FlangCommon.cmake`.

 * Some header files are included from `.c` files.
   `#include <flang/Common/c-or-cpp.h>` can help to support C++ and C.

 * Global declarations may need to be annotated using definitions from
   `api-attrs.h`.

 * The `Runtime` component is header-only.
