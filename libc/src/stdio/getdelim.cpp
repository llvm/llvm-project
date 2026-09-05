#include "src/stdio/getdelim.h"

#include "hdr/types/FILE.h"
#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/File/file.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/stdio/inline_getline.h"

namespace LIBC_NAMESPACE_DECL {

  LLVM_LIBC_FUNCTION(ssize_t, getdelim, (char **__restrict lineptr, size_t *__restrict n,
                    int delimiter, ::FILE *__restrict stream)) {
  return LIBC_NAMESPACE::__getline(lineptr, n, delimiter, stream);
}
} // namespace LIBC_NAMESPACE_DECL
